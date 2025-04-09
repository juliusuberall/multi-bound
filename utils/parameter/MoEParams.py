from flax import struct

# Network layer overview
## Gate (data input dimension, ... , hidden, ... , number of expert networks) 
## -> returns probability distribution of activated experts
## Expert (data input dimension, ... , hidden, ..., data output dimension)
## -> returns expert wise predicition

class MoEParams(struct.PyTreeNode):
    pass

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