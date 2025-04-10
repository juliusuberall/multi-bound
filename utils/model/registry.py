from utils.model.MLP import MLP
from utils.model.MoE import MoE

# Look-up table for reading model configurations
model_registry = {
 "mlp": MLP,
 "moe": MoE,
}