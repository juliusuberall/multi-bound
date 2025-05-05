import time 
import jax.numpy as jnp
from jax import random, jit
from flax import struct
import jax

# This test evaluates the average speed performance between different JIT scenarios involving:
#
# 1. Baseline speed of NOT JIT compiled function
# 2. Speed of JIT compile function
# 3. Speed of static class method which is JIT compiled
# 4. Speed of static class method which is JIT compiled and called with a custom PyTree parameters which update
#
# -> So far results indicate that there is no speed difference between 2. and 3.
# -> Seems like JIT can reduce time to 10% of original function evaluation time

# ------------------------------------------------------------------------------------
# Define functions to evaluate
## 1.
def activation(x):
    return x**2 + x + 5
## 2.
activation_jit = jit(activation)
## 3.
class Model():
    @staticmethod
    @jax.jit
    def activation(x):
        print("⚡ Compiling...")
        return x**2 + x + 5
## 4.
class Params(struct.PyTreeNode):
    params: jax.Array
class Model2():
    def __init__(self, p):
        self.p = p
    @staticmethod
    @jax.jit
    def activation(x:Params):
        print("⚡ Compiling...")
        return x.params**2 + x.params + 5

# Baseline variables
rkey = random.PRNGKey(28)
x = random.normal(rkey,(1000))
reps = 10000
print(jax.devices())

# ------------------------------------------------------------------------------------
# Measure speed
## Crucial for JAX to use .block_until_ready() since
## JAX uses asynchronous dispatching and time evaluation 
## would be wrong without this call
### https://docs.jax.dev/en/latest/async_dispatch.html
# 1.
s = time.time()
for _ in range(reps):
    activation(x).block_until_ready()
e = time.time()

# 2.
## JIT warm up to trace and compile function
### https://docs.jax.dev/en/latest/jit-compilation.html#jit-compiling-a-function
activation_jit(x).block_until_ready()
sj = time.time()
for _ in range(reps):
    activation_jit(x).block_until_ready()
ej = time.time()

# 3.
Model.activation(x).block_until_ready()
sjc = time.time()
for _ in range(reps):
    Model.activation(x).block_until_ready()
ejc = time.time()

# 4.
p = Params( params = x)
model = Model2(Params(params = x))
Model2.activation(model.p).block_until_ready()
sjcp = time.time()
for _ in range(reps):
    model.p = Params(Model2.activation(model.p).block_until_ready())
ejcp = time.time()

# ------------------------------------------------------------------------------------
print(f"############################### Speed Results ################################")
print(f"-> Base Avg. time: {((e - s)/reps * 1000000):.0f} microseconds")
print(f"-> JIT Avg. time: {((ej - sj)/reps * 1000000):.0f} microseconds")
print(f"-> Class JIT Avg. time: {((ejc - sjc)/reps * 1000000):.0f} microseconds")
print(f"-> Class JIT PyTree Params Avg. time: {((ejcp - sjcp)/reps * 1000000):.0f} microseconds")