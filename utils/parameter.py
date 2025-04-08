from flax import struct
from utils.registry import *
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

        with open(f"{model_dir["model_params_dir"]+'/'+path}.json", "w") as f:
            json.dump(serializable_p, f)