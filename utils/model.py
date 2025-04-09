from abc import ABC, abstractmethod
from utils.sampler import *
from utils.parameter import *
from utils.registry import *
import jax.numpy as jnp
import jax
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

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
            params = self.init_layer(
                [sampler.x_dim] + config["hidden_layer"] + [sampler.y_dim],
                key
            )
        )
    
    @staticmethod
    @jax.jit
    def forward(p:MLPParams, x):
        for W, b in p.params:
            x = jax.nn.leaky_relu(jnp.dot(x, W) + b, 0.01)
        return x
    
    @staticmethod
    @jax.jit
    def loss(p:MLPParams, x, y):
        preds = jax.vmap(lambda x: MLP.forward(p, x))(x)
        return jnp.mean((preds - y) ** 2)
    
    @staticmethod
    def signal_inference_solid_IMG(p:MLPParams, sampler:RGBAImageSampler):
        """
        Inferes non-transparent, solid image signal.
        Important since the model should only be fit to the solid region of the image.

        Args
        ----------
        p :
            The model parameters
        sampler :
            The sampler used for training the model

        Returns
        ----------
        reconstructed_signal :
            The full reconstructed image signal. Values in range 0.0 - 1.0 .
        y :
            Ground truth signal
        """
        # Inference of all pixel from image
        x, y = sampler.inference_sample_solid()
        reconstructed_signal = jnp.clip(
            jax.vmap(lambda x: MLP.forward(p, x))(x),
            0,
            1
        )
        return reconstructed_signal, y

    @staticmethod
    def full_signal_inference_IMG(p:MLPParams, sampler:RGBAImageSampler):
        """
        Inferes the full image signal including transparent regions.

        Args
        ----------
        p :
            The model parameters
        sampler :
            The sampler used for training the model

        Returns
        ----------
        reconstructed_signal :
            The full reconstructed image signal. Values in range 0.0 - 255.0 .
        """
        # Inference of all pixel from image
        reconstructed_signal = jnp.clip(
            jax.vmap(lambda x: MLP.forward(p, x))(sampler.inference_sample()) * 255,
            0,
            255
        )
        return reconstructed_signal

    @staticmethod
    def save_full_signal_inference_IMG(p:MLPParams, sampler:RGBAImageSampler, model_name):
        """
        Reconstructs the full image signal and saves the image. Also inferes transparent regions.
        Saves result with and without transparent image regions.

        Args
        ----------
        p :
            The model parameters
        sampler :
            The sample used for training the model
        model_name :
            The name of the model architecture

        Returns
        ----------
        path_full :
            The path to the full reconstructed image signal
        path_mask :
            The path to the masked reconstructed image signal
        """
        # Create reconstruction directory and save masked and non-masked image
        if not os.path.isdir(dir_registry["reconstruction_dir"]):
            os.makedirs(dir_registry["reconstruction_dir"])
        ## Non-masked
        reconstructed_signal = MLP.full_signal_inference_IMG(p, sampler)
        path_full = f'{dir_registry["reconstruction_dir"]}/{model_name}_full.png'
        Image.fromarray(np.array(reconstructed_signal).astype(np.uint8)).save(path_full)
        
        ## Masked
        path_mask = f'{dir_registry["reconstruction_dir"]}/{model_name}_masked.png'
        reconstructed_signal = jnp.concat((reconstructed_signal, jnp.expand_dims(sampler.alpha, axis=-1)*255), axis=-1)
        Image.fromarray(np.array(reconstructed_signal).astype(np.uint8)).save(path_mask)

        return path_full, path_mask

    def flatten_func(obj):
        children = (obj.params)
        aux_data = (obj.learning_rate,)
        return (children, aux_data)
    
    def unflatten_func(aux_data, children):
        obj = object.__new__(MLP)
        obj.params = children
        obj.learning_rate, = aux_data
        return obj

    def serialize(self, path):
        """
        Serializes the parameters. 
        Originally the paramaters are in a JIT compatable structure which can not be directly be serialized.

        Args
        ----------
        path :
            Path to serialize to.
        """
        self.params.serialize(path)

    @staticmethod
    def deserialize(path:str) -> MLPParams:
        """
        Deserializes the parameters.

        Args
        ----------
        path :
            Parameter file to deserialize to.
        """
        p = MLPParams.deserialize(path)
        return p


# Register class as jit compilable
jax.tree_util.register_pytree_node(
    MLP,
    MLP.flatten_func,
    MLP.unflatten_func
)

# Look-up table for reading model configurations
model_registry = {
 "mlp": MLP,
}