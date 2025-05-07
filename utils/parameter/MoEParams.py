from utils.registry import *
from utils.parameter.BaseParams import *
import jax.numpy as jnp
import json
import os
from typing import Self

# Network layer overview
## Gate (data input dimension, ... , hidden, ... , number of expert networks) 
## -> returns probability distribution of activated experts
## Expert (data input dimension, ... , hidden, ..., data output dimension)
## -> returns expert wise predicition

## Experts list layout
# experts/
# ├── expert_0/
# │   ├── layer_0/
# │   │   ├── W  (JAX array, 2D)
# │   │   └── b  (JAX array, 1D)
# │   ├── layer_1/
# │   │   ├── W  (JAX array, 2D)
# │   │   └── b  (JAX array, 1D)
# │   └── ...
# ├── expert_1/
# │   ├── layer_0/
# │   │   ├── W
# │   │   └── b
# │   ├── layer_1/
# │   │   ├── W
# │   │   └── b
# │   └── ...
# └── ...

class MoEParams(BaseParams):
    """
    <summary>
        Base MoE parameter structure.
    </summary>
    """
    gate: list = None # Gate network parameters
    experts: list = None # Expert networks parameters

    def serialize(self, path:str="parameters"):
        # Final dictionary for gate and experts
        all_p = {}

        # Loop over all EXPERT networks in MoE and save in 
        # list(network) of list(layer) of lists(weights & bias)
        all_experts = []
        for expert in self.experts:
            p = []
            for W, b in expert:
                p.append([W.tolist(), b.tolist()])
            all_experts.append(p)
        
        # Loop over all GATE networks in MoE and save in 
        # list(network) of list(layer) of lists(weights & bias)
        gate = []
        for W, b in self.gate:
            gate.append([W.tolist(), b.tolist()])

        # Add to dictionary to serialize
        all_p['experts'] = all_experts
        all_p['gate'] = gate

        # Create models directory and save parameters
        if not os.path.isdir(dir_registry["model_params_dir"]):
            os.makedirs(dir_registry["model_params_dir"])
        with open(f"{dir_registry["model_params_dir"]+'/'+path}.json", "w") as f:
            json.dump(all_p, f)
    
    @staticmethod
    def deserialize(file:str) -> Self:
        # Expects a dictionary to deserialize. Will use the order of expert
        # parameters to set the parameters in the MoEParams struct.
        p = []
        with open(file) as f:
            d = json.load(f)

            p = MoEParams(
                # Set gate parameters 
                gate = [(jnp.array(W), jnp.array(b)) for W, b in d['gate']],

                # Set expert parameters as list of experts
                experts = [[(jnp.array(W), jnp.array(b)) for W, b in expert] 
                    for expert in d['experts']]
            )
        return p
