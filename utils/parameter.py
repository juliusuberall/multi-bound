from flax import struct

class MLPParams(struct.PyTreeNode):
    mlp: list