import time
import jax
import numpy as np
from jax import random
from utils.model.MLP import *
from utils.model.BaseModel import BaseModel

# This test evaluates the speed for different queries as per Neural Bounding (Liu et. al)
# Appendix table 6. The paper used a workstation with an RTX3090 GPU and Intel i9-12900K CPU.
#
# 0. 2D point [32,32]
#
# -> Since the workstation from the paper is high-spec it runs really fast.

# ------------------------------------------------------------------------------------
# General setup
rkey = jax.random.PRNGKey(29)
query_size = 100000 # "10 million samples per run" as per paper 
point2d = random.normal(rkey,(query_size, 2))
output_dim = 1

# 0. >>>>>>>>>>>>>>>
p0 = MLPParams(
    params = BaseModel.init_layer(
        [2] + [32,32] + [output_dim],
        rkey
    )
)

# ------------------------------------------------------------------------------------
# Measure speed
reps = 10000 # "10,000 independent runs" as per paper 
batch_size = 128

# 0.
point2d_batched = DataSampler.batch_generator(point2d, batch_size)
MLP.forward(p0, point2d_batched[0,...]).block_until_ready()
s0 = time.time()
for _ in range(reps):
    out = jax.vmap(lambda x: MLP.forward(p0, x))(point2d_batched)
    out.block_until_ready()
e0 = time.time()

# ------------------------------------------------------------------------------------
print(f"############### Speed Results - {reps} Reps. ################")
print(f"(0) -> Avg. time: {((e0 - s0)/reps * 1000):.0f}ms")