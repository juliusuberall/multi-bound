from abc import ABC, abstractmethod
from flax import struct

class BaseParams(struct.PyTreeNode):
    """
    <summary>
        Base model parameter class which is inherited by every model parameter. This is an abstract class.
    </summary>
    """
    
    @abstractmethod
    def serialize():
        """
        Serializes the parameters. 
        Originally the paramaters are in a list and tuple wrapped JAX arrays which which can not be directly be serialized.

        Args
        ----------
        path :
            Path to serialize to.
        """
        pass

    @abstractmethod
    def deserialize(): 
        """
        Deserializes the parameters into List of Tuples with JAX arrays for weights and bias.

        Args
        ----------
        path :
            Parameter file to deserialize to.

        Returns
        ----------
        p :
            Model parameters
        """
        pass