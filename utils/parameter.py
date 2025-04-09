from flax import struct
from utils.registry import *
import jax.numpy as jnp
import os
import json

class MLPParams(struct.PyTreeNode):
    params: list

    def serialize(self, path:str="parameters"):
        """
        Serializes the parameters. 
        Originally the paramaters are in a JIT compatable structure which can not be directly be serialized.

        Args
        ----------
        path :
            Path to serialize to.
        """
        serializable_p = []
        for W, b in self.params:
            serializable_p.append([W.tolist(), b.tolist()])
        
        # Create models directory and save parameters
        if not os.path.isdir(dir_registry["model_params_dir"]):
            os.makedirs(dir_registry["model_params_dir"])
        with open(f"{dir_registry["model_params_dir"]+'/'+path}.json", "w") as f:
            json.dump(serializable_p, f)
    
    @staticmethod
    def deserialize(file:str):
        """
        Deserializes the parameters into List of Tuples with JAX arrays for weights and bias.

        Args
        ----------
        path :
            Parameter file to deserialize to.
        """
        p = []
        with open(file) as f:
            d = json.load(f)

            # Convert back into list of tuples storing JAX arrays for weights and bias
            for W, b in d:
                p.append((jnp.array(W), jnp.array(b)))

        return MLPParams(params = p)