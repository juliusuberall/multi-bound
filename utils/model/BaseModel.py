from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax

class BaseModel(ABC):
    """
    <summary>
        Base model class which is inherited by every model architecture. This is an abstract class.
    </summary>
    """

    def init_layer(self, layer_dims, key):
        """
        Initalize parameters of network (weights and biases)

        Args
        ----------
        layer_dims :
            The layer dimensions. Including input, hidden and output layer dims.
        key :
            Jax random key used for sampling.
        """
        p = []
        keys = jax.random.split(key, len(layer_dims))
        for k, (n_in, n_out) in zip(keys, zip(layer_dims[:-1], layer_dims[1:])):
            W = jax.random.normal(k, (n_in, n_out)) * 0.1
            b = jnp.zeros((n_out,))
            p.append((W, b))
        return p
    
    @abstractmethod
    def forward(): pass

    @abstractmethod
    def loss(): pass

    @abstractmethod
    def full_signal_inference_IMG(): pass

    # Implement functions to allow jit handeling for custom class
    # https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html
    @abstractmethod
    def flatten_func(): pass
    @abstractmethod
    def unflatten_func(): pass

    @abstractmethod
    def serialize(): pass

    @abstractmethod
    def deserialize(): pass