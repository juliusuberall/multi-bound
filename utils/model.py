from abc import ABC, abstractmethod
from utils.parameter import MLPParams
from utils.sampler import *
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# Base class for sampling training data
# Relevant when changing training data format
class BaseModel(ABC):

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
    def flatten_func(): pass

    @abstractmethod
    def unflatten_func(): pass

class MLP(BaseModel):

    def __init__(self, config, sampler:DataSampler, key):
        """
        Instantiates vanilla Multilayer Perceptron for the given sampler.

        Args
        ----------
        config :
            The model configuration.
        sampler :
            The data sampler used for training the model.
        key :
            Jax random key used for sampling.
        """
        self.learning_rate = config["learning_rate"]
        self.params = self.init_layer(
                [sampler.x_dim] + config["hidden_layer"] + [sampler.y_dim],
                key
            )
    
    @staticmethod
    @jax.jit
    def forward(p, x):
        for W, b in p:
            x = jax.nn.leaky_relu(jnp.dot(x, W) + b, 0.01)
        return x
    
    @staticmethod
    @jax.jit
    def loss(p, x, y):
        preds = jax.vmap(lambda x: MLP.forward(p, x))(x)
        return jnp.mean((preds - y) ** 2)
    
    # Implement functions to allow jit handeling for custom class
    # https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html
    def flatten_func(obj):
        children = (obj.params)
        aux_data = (obj.learning_rate,)
        return (children, aux_data)
    
    def unflatten_func(aux_data, children):
        obj = object.__new__(MLP)
        obj.params = children
        obj.learning_rate, = aux_data
        return obj
    
# Look-up table for reading model configurations
model_registry = {
 "mlp": MLP,
}