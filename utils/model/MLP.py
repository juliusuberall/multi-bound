from utils.sampler import *
from utils.parameter.MLPParams import MLPParams
from utils.registry import *
from utils.model.BaseModel import BaseModel
import jax.numpy as jnp
import jax
from PIL import Image
import numpy as np
import os

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

    def serialize(self, path):
        self.params.serialize(path)

    @staticmethod
    def deserialize(path:str) -> MLPParams:
        p = MLPParams.deserialize(path)
        return p
