from utils.model.MLP import MLP
from utils.model.MoE import MoE
from utils.model.MoEH import MoEH

# Look-up table for reading model configurations
model_registry = {
 "mlp": MLP,
 "moe": MoE, 
 "moeH": MoEH,
}