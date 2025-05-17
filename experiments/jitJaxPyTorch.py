import jax
import time
import numpy as n
import torch
import torch.nn as nn
from jax import random
from utils.parameter.MLPParams import *
from utils.model.MLP import *
from utils.model.BaseModel import *

# This test evaluates inference speed between JAX and Pytorch.
# Evaluates with ray query since neural bounding speeds is using such.
# 6 dimensional input vector and 1 dimensional output vector.
#
# We use the Neural Bounding MLP architecture for inference comparison from table 6.
#
# 0. JAX JIT
# 1. PyTorch
#
# -> Seems like the Pytorch MLP runs slighlty faster than JAX
# -> Somehow it seems that both runs significantly faster than in the Neural Bounding Paper...
#    Not sure if that is due to difference in machine (in theory neural bounding machine is much better).

# ------------------------------------------------------------------------------------
# Set up model query
rkey = jax.random.PRNGKey(28)
j_input = random.normal(rkey,(1000, 6))
p_input = torch.from_numpy(np.asarray(j_input))

output_dim = 1
print(f"Query Dimension: {j_input.shape[1]}")

# ------------------------------------------------------------------------------------
# Implementation of architectures to evaluate
# 0. >>>>>>>>>>>>>>>
# MLP configuration and setup
config = {
    "hidden_layer": [50,50]
}
mlp_p = MLPParams(
    params = BaseModel.init_layer(
        [j_input.shape[1]] + config["hidden_layer"] + [output_dim],
        rkey
    )
)

# 1. >>>>>>>>>>>>>>>
# Define PyTorch model
class PyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear( j_input.shape[1], 50),
            nn.LeakyReLU(),

            nn.Linear( 50, 50),
            nn.LeakyReLU(),

            nn.Linear( 50, output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

# ------------------------------------------------------------------------------------
# Measure speed
reps = 1000

# jax.config.update("jax_disable_jit", False) # True = JIT off

# 0.
MLP.forward(mlp_p, j_input).block_until_ready()
s0 = time.time()
for _ in range(reps):
    MLP.forward(mlp_p, j_input).block_until_ready()
e0 = time.time()

# 1.
py_mlp = PyMLP()
py_mlp.eval()
s1 = time.time()
for _ in range(reps):
    py_mlp(p_input)
e1 = time.time()


# ------------------------------------------------------------------------------------
print(f"############### Speed Results - {reps} Reps. ################")
print(f"(0) -> Avg. time: {((e0 - s0)/reps * 1000000):.0f}μs")
print(f"(1) -> Avg. time: {((e1 - s1)/reps * 1000000):.0f}μs")