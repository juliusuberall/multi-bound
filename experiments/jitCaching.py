import jax
import jax.numpy as jnp

# JIT compiling test that identifies when a function is recompiled. This can happen
# when the input shapes change or the argument type changes significantly. Such recompilation
# can be identified with Python side-effects as they dont appear in the JIT compiled cache of the function.
# Print() statements and python side-effects and will only print during JAX trace of a function 
# and highlight once a compilation is happening.
## https://docs.jax.dev/en/latest/jit-compilation.html#how-jax-transformations-work
## https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions 

# ------------------------------------------------------------------------------------
@jax.jit
def f(x):
    print("⚡ Compiling...")
    return x + 1

x1 = jnp.ones((2, 2), dtype=jnp.float32)
x2 = jnp.ones((4, 4), dtype=jnp.float32)
x3 = jnp.ones((2, 2), dtype=jnp.float64)
x4 = jnp.ones((2, 2), dtype=jnp.int32)

f(x1)  # COMPILES
f(x1)  # Uses cache
f(x2)  # RECOMPILES (shape changed)
f(x3)  # Uses cache (even though dtype changed ?)
f(x4)  # RECOMPILES (dtype changed)