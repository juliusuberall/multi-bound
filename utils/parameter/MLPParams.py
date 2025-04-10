from utils.registry import *
from utils.parameter.BaseParams import *
import jax.numpy as jnp
import os
import json

class MLPParams(BaseParams):
    """
    <summary>
        Simple MLP paramater.
    </summary>
    """
    params: list

    def serialize(self, path:str="parameters"):
        serialized_p = []
        for W, b in self.params:
            serialized_p.append([W.tolist(), b.tolist()])
        
        # Create models directory and save parameters
        if not os.path.isdir(dir_registry["model_params_dir"]):
            os.makedirs(dir_registry["model_params_dir"])
        with open(f"{dir_registry["model_params_dir"]+'/'+path}.json", "w") as f:
            json.dump(serialized_p, f)
    
    @staticmethod
    def deserialize(file:str):
        p = []
        with open(file) as f:
            d = json.load(f)

            # Convert back into list of tuples storing JAX arrays for weights and bias
            for W, b in d:
                p.append((jnp.array(W), jnp.array(b)))

        return MLPParams(params = p)