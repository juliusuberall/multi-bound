from utils.registry import *
from utils.parameter.BaseParams import *
import jax.numpy as jnp
import json
import os

# Network layer overview
## Gate (data input dimension, ... , hidden, ... , number of expert networks) 
## -> returns probability distribution of activated experts
## Expert (data input dimension, ... , hidden, ..., data output dimension)
## -> returns expert wise predicition

class MoEParams(BaseParams):
    """
    <summary>
        Base MoE parameter structure.
    </summary>
    """
    def serialize(self, path:str="parameters"):
        serialized_p = []

        # Loop over all gate and expert networks in MoE and save in 
        # list(network) of list(layer) of lists(weights & bias)
        for n in self.__dataclass_fields__:
            sub_network_p = []
            for W, b in self.__getattribute__(n):
                sub_network_p.append([W.tolist(), b.tolist()])
            serialized_p.append(sub_network_p)
        
        # Create models directory and save parameters
        if not os.path.isdir(dir_registry["model_params_dir"]):
            os.makedirs(dir_registry["model_params_dir"])
        with open(f"{dir_registry["model_params_dir"]+'/'+path}.json", "w") as f:
            json.dump(serialized_p, f)
    
    @staticmethod
    def deserialize(file:str):
        # Expects the serialized MoE to follow the standard -> [gate, 1st expert, ..., nth expert]
        # and match the order in the corresponding MoEParamsXXX struct
        p = []
        with open(file) as f:
            d = json.load(f)

            # Determine the number of experts and select correct MoEParams struct
            n_experts = len(d) - 1 
            moe_params = moe_params_registry[n_experts]()

            # Convert back into list of tuples storing JAX arrays for weights and bias
            n = moe_params.__dataclass_fields__
            for i, key in enumerate(n):
                ## Load from json and insert in each network
                p = []
                for W, b in d[i]:
                    p.append((jnp.array(W), jnp.array(b)))

                ## **{x:y} transforms into keyword argument x=y
                moe_params = moe_params.replace(**{key: p})

        return moe_params

class MoEParams2E(MoEParams):
    """
    <summary>
        MoE paramater struct for 1 gate and 2 expert networks.
    </summary>
    """
    gate: list = None
    expert1: list = None
    expert2: list = None

class MoEParams4E(MoEParams):
    """
    <summary>
        MoE paramater struct for 1 gate and 4 expert networks.
    </summary>
    """
    gate: list = None
    expert1: list = None
    expert2: list = None
    expert3: list = None
    expert4: list = None

# MoE parameter registry to identify which struct layout to use
moe_params_registry = {
    2 : MoEParams2E,
    4 : MoEParams4E
}