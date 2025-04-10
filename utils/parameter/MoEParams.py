from flax import struct
from utils.registry import *
from utils.parameter.BaseParams import *
import json
import os

# Network layer overview
## Gate (data input dimension, ... , hidden, ... , number of expert networks) 
## -> returns probability distribution of activated experts
## Expert (data input dimension, ... , hidden, ..., data output dimension)
## -> returns expert wise predicition

class MoEParams2E(BaseParams):
    """
    <summary>
        MoE paramater struct for 1 gate and 2 expert networks.
    </summary>
    """
    gate: list = None
    expert1: list = None
    expert2: list = None

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


class MoEParams4E(BaseParams):
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