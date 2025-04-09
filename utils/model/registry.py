from utils.model.MLP import MLP
from utils.model.MoE import MoE
import jax

# Register class as jit compilable
jax.tree_util.register_pytree_node(
    MLP,
    MLP.flatten_func,
    MLP.unflatten_func
)

# Look-up table for reading model configurations
model_registry = {
 "mlp": MLP,
 "moe": MoE,
}