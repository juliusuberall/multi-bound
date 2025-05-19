from utils.DataSampler import *
from utils.parameter.MLPParams import MLPParams
from utils.registry import *
from utils.model.BaseModel import BaseModel
import jax.numpy as jnp
import jax

class MLP(BaseModel):
    """
    <summary>
        Vanilla MLP implementation that is composed out of a single network.
    </summary>
    """
    
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
        self.params = MLPParams(
            params = BaseModel.init_layer(
                [sampler.x_dim] + config["hidden_layer"] + [sampler.y_dim],
                key
            )
        )
    
    @staticmethod
    @jax.jit
    def fastforward(p:MLPParams, x):
        for W, b in p.params:
            x = jax.nn.leaky_relu(jnp.dot(x, W) + b, 0.01)
        return x

    @staticmethod
    @jax.jit
    def forward(p:MLPParams, x):
        return jax.vmap(lambda x: MLP.fastforward(p, x))(x)
    
    @staticmethod
    @jax.jit
    def loss(p:MLPParams, x, y):
        preds = MLP.forward(p, x)
        return jnp.mean((preds - y) ** 2)

    def serialize(self, path):
        self.params.serialize(path)

    @staticmethod
    def deserialize(path:str) -> MLPParams:
        p = MLPParams.deserialize(path)
        return p
