import jax
import jax.numpy as jnp
import time
from jax import random
from utils.parameter.MoEParams import *
from utils.model.MoE import *
from utils.DataSampler import RGBAImageSampler

# This test evaluates the speed of sparse gated MoE forward implementations.
# All scenarios use the same MoE parameter PyTree and topK. Increasingly the scenarios 
# make additional changes to the forward(call). So all changes made in previous scenarios 
# apply to the selected sencario as well within a bracket from 0-10, 10-20 and 20-30.
#
# 0. Baseline implementation
# 1. Lambda expressions to run and map input to experts is defined static outside of function
# 2. Removes nested vmap and for-loops top-k times and computes with single vmapped-switch
# 10. Hard-coded 2 experts
# 11. Removes recalculation of topK weights to sum to 1
# 20. Hard-coded all experts
#
# -> Inference speed does not scale linearly with model complexity and rather gets much slower 
#    as the model becomes larger
# -> Main bottleneck seems to be the implementation in 0. and 1. because even if we run all experts
#    like in 20. its still significantly faster then 0. and 1. which in theory have half the model
#    queries only.
# -> 0. and 1. scale badly with increasing number of top-k and speed drops fast.

# ------------------------------------------------------------------------------------
# Model configuration and setup
config = {
    "n_experts" : 4,
    "expert_hidden_layer": [16,16,16],
    "gate_hidden_layer": [8,8,8],
    "learning_rate": 0.01,
}
top_k = 2

# General setup to utilize constructor for parameter creation
rkey = jax.random.PRNGKey(28)
sampler = RGBAImageSampler('data/img/dolphin_color.png')

# Set up MoE model with parameters used for all scenarios
moe = MoE(config, sampler, rkey)

# Set up model query
input = random.normal(rkey,(1000, 2))

# ------------------------------------------------------------------------------------
# Implementation of functions to evaluate
# 0. >>>>>>>>>>>>>>>
@jax.jit
def forward0(p:MoEParams, x):
    # Get top K indicies
    ## gate output shape : [batchsize, number of experts
    gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)
    gate_probs , idx = jax.lax.top_k(gate, top_k)

    # Define expert evaluation branches dynamically 
    ## Uses lambda to define local, unnamed python functions which represent 
    ## the execution of the correct expert branch for an input
    branches = tuple(
        (lambda expert_params: (lambda x: MoE.forward_expert(expert_params, x)))(expert)
        for expert in p.experts
    )
    
    # Forward through activated experts
    @jax.jit
    def expert_leaf_branching(x, i):
        return jax.lax.switch(i, branches, x)
    
    @jax.jit
    def expert_branching(x, i):
        return jax.vmap(lambda i: expert_leaf_branching(x, i))(i)

    expert_out = jax.vmap(expert_branching)(x, idx)

    # Compute weighted sum based on recalculcated gate probabilities, such that 
    # selected experts sum to 1. If all experts of MoE used, weights remain the same.
    gate_probs /= jnp.expand_dims(jnp.sum(gate_probs, axis=-1), axis=-1)
    expert_out *= jnp.expand_dims(gate_probs, axis=-1)
    expert_out = jnp.sum(expert_out, axis=-2)

    return expert_out


# 1. >>>>>>>>>>>>>>>
branches = tuple(
    (lambda expert_params: (lambda x: MoE.forward_expert(expert_params, x)))(expert)
    for expert in moe.params.experts
)
@jax.jit
def forward1(p:MoEParams, x):
    ## gate output shape : [batchsize, number of experts
    gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)
    gate_probs , idx = jax.lax.top_k(gate, top_k)
    
    # Forward through activated experts
    @jax.jit
    def expert_leaf_branching(x, i):
        return jax.lax.switch(i, branches, x)
    
    @jax.jit
    def expert_branching(x, i):
        return jax.vmap(lambda i: expert_leaf_branching(x, i))(i)

    expert_out = jax.vmap(expert_branching)(x, idx)

    # Compute weighted sum based on recalculcated gate probabilities, such that 
    # selected experts sum to 1. If all experts of MoE used, weights remain the same.
    gate_probs /= jnp.expand_dims(jnp.sum(gate_probs, axis=-1), axis=-1)
    expert_out *= jnp.expand_dims(gate_probs, axis=-1)
    expert_out = jnp.sum(expert_out, axis=-2)

    return expert_out


# 2. >>>>>>>>>>>>>>>
branches = tuple(
    (lambda expert_params: (lambda x: MoE.forward_expert(expert_params, x)))(expert)
    for expert in moe.params.experts
)
jax.jit
def expert(x, idx):
    return jax.lax.switch(idx, branches, x)
@jax.jit
def forward2(p:MoEParams, x):
    ## gate output shape : [batchsize, number of experts]
    gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)
    gate_probs , idx = jax.lax.top_k(gate, top_k)

    out = []
    for k in range(top_k):
        out.append(jax.vmap(expert)(x, idx[...,k]))
    expert_out = jnp.stack(out, axis=1)

    # Compute weighted sum based on recalculcated gate probabilities, such that 
    # selected experts sum to 1. If all experts of MoE used, weights remain the same.
    gate_probs /= jnp.expand_dims(jnp.sum(gate_probs, axis=-1), axis=-1)
    expert_out *= jnp.expand_dims(gate_probs, axis=-1)
    expert_out = jnp.sum(expert_out, axis=-2)

    return expert_out


# 10. >>>>>>>>>>>>>>>
branches = tuple(
    (lambda expert_params: (lambda x: MoE.forward_expert(expert_params, x)))(expert)
    for expert in moe.params.experts
)
@jax.jit
def forward10(p:MoEParams, x):
    # Get top K indicies
    ## gate output shape : [batchsize, number of experts
    gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)
    gate_probs , idx = jax.lax.top_k(gate, top_k)

    ### Hard coded first two experts
    a = jax.vmap(branches[0])(x)
    b = jax.vmap(branches[1])(x)
    expert_out = jnp.stack((a,b), axis=1)

    # Compute weighted sum based on recalculcated gate probabilities, such that 
    # selected experts sum to 1. If all experts of MoE used, weights remain the same.
    gate_probs /= jnp.expand_dims(jnp.sum(gate_probs, axis=-1), axis=-1)
    expert_out *= jnp.expand_dims(gate_probs, axis=-1)
    expert_out = jnp.sum(expert_out, axis=-2)

    return expert_out


# 11. >>>>>>>>>>>>>>>
branches = tuple(
    (lambda expert_params: (lambda x: MoE.forward_expert(expert_params, x)))(expert)
    for expert in moe.params.experts
)
@jax.jit
def forward11(p:MoEParams, x):
    # Get top K indicies
    ## gate output shape : [batchsize, number of experts
    gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)
    gate_probs , idx = jax.lax.top_k(gate, top_k)

    ### Hard coded first two experts
    a = jax.vmap(branches[0])(x)
    b = jax.vmap(branches[1])(x)
    expert_out = jnp.stack((a,b), axis=1)

    # Compute weighted sum based on recalculcated gate probabilities, such that 
    # selected experts sum to 1. If all experts of MoE used, weights remain the same.
    expert_out *= jnp.expand_dims(gate_probs, axis=-1)
    expert_out = jnp.sum(expert_out, axis=-2)

    return expert_out


# 20. >>>>>>>>>>>>>>>
branches = tuple(
    (lambda expert_params: (lambda x: MoE.forward_expert(expert_params, x)))(expert)
    for expert in moe.params.experts
)
@jax.jit
def forward20(p:MoEParams, x):
    # Get top K indicies
    top_k = config["n_experts"]
    ## gate output shape : [batchsize, number of experts
    gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)
    gate_probs , idx = jax.lax.top_k(gate, top_k)

    ### Hard coded first two experts
    a = jax.vmap(branches[0])(x)
    b = jax.vmap(branches[1])(x)
    c = jax.vmap(branches[2])(x)
    d = jax.vmap(branches[3])(x)
    expert_out = jnp.stack((a,b,c,d), axis=1)

    # Compute weighted sum based on recalculcated gate probabilities, such that 
    # selected experts sum to 1. If all experts of MoE used, weights remain the same.
    expert_out *= jnp.expand_dims(gate_probs, axis=-1)
    expert_out = jnp.sum(expert_out, axis=-2)

    return expert_out

# ------------------------------------------------------------------------------------
# Measure speed
reps = 1000

# 0.
forward0(moe.params, input).block_until_ready()
s0 = time.time()
for _ in range(reps):
    forward0(moe.params, input).block_until_ready()
e0 = time.time()

# 1.
forward1(moe.params, input).block_until_ready()
s1 = time.time()
for _ in range(reps):
    forward1(moe.params, input).block_until_ready()
e1 = time.time()

# 2.
forward2(moe.params, input).block_until_ready()
s2 = time.time()
for _ in range(reps):
    forward2(moe.params, input).block_until_ready()
e2 = time.time()

# 10.
forward10(moe.params, input).block_until_ready()
s10 = time.time()
for _ in range(reps):
    forward10(moe.params, input).block_until_ready()
e10 = time.time()

# 11.
forward11(moe.params, input).block_until_ready()
s11 = time.time()
for _ in range(reps):
    forward11(moe.params, input).block_until_ready()
e11 = time.time()

# 20.
forward20(moe.params, input).block_until_ready()
s20 = time.time()
for _ in range(reps):
    forward20(moe.params, input).block_until_ready()
e20 = time.time()

# ------------------------------------------------------------------------------------
print(f"############### Speed Results - {reps} Reps. ################")
print(f"(0) -> Avg. time: {((e0 - s0)/reps * 1000000):.0f}μs")
print(f"(1) -> Avg. time: {((e1 - s1)/reps * 1000000):.0f}μs")
print(f"(2) -> Avg. time: {((e2 - s2)/reps * 1000000):.0f}μs")
print(f"(10) -> Avg. time: {((e10 - s10)/reps * 1000000):.0f}μs")
print(f"(11) -> Avg. time: {((e11 - s11)/reps * 1000000):.0f}μs")
print(f"(20) -> Avg. time: {((e20 - s20)/reps * 1000000):.0f}μs")