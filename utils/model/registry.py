from utils.model.MLP import MLP
from utils.model.MoE import MoE
from utils.model.MoEH import MoEH

# Look-up table for reading model configurations
model_registry = {
 "mlp": MLP,
 "moe": MoE, 
 "moeH": MoEH,
}

# Training settings
train_set = {
    "plateau_depth" : 3, # Number of previous losses which are averaged to determine convergence plateau
    "epsilon" : 1e-3,
    "batch_size" : 1024,
}