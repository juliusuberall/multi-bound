from utils.registry import *
from utils.DataSampler import *
from utils.parameter.BaseParams import *
from abc import ABC, abstractmethod
from PIL import Image
import os
import numpy as np
import jax.numpy as jnp
import jax

class BaseModel(ABC):
    """
    <summary>
        Base model class which is inherited by every model architecture. This is an abstract class.
    </summary>
    """

    def init_layer(layer_dims, key):
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
        Deserializes the parameters.

        Args
        ----------
        path :
            Parameter file to deserialize to.
        """
        pass

    @staticmethod
    def full_signal_inference_IMG(p:BaseParams, sampler:RGBAImageSampler, model_type):
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
            The full reconstructed image signal. Values in range 0.0 - 1.0 .
        """
        # Get all image coordinates and flatten into 2D array for vmap
        cor = sampler.inference_sample()
        cor_flat = cor.reshape(-1, cor.shape[-1])
                
        # Inference of all pixel from image
        reconstructed_signal = jnp.clip(
            model_type.forward(p, cor_flat),
            0,
            1
        )

        # Reshape into [width, height, channels]
        return reconstructed_signal.reshape(cor.shape[0], cor.shape[1], reconstructed_signal.shape[-1])

    @staticmethod
    def save_full_signal_inference_IMG(p:BaseParams, sampler:RGBAImageSampler, model_name, model_type):
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
        # Model inference 
        reconstructed_signal = model_type.full_signal_inference_IMG(p, sampler, model_type) * 255

        # Create reconstruction directory and save masked and non-masked image
        if not os.path.isdir(dir_registry["reconstruction_dir"]):
            os.makedirs(dir_registry["reconstruction_dir"])

        ## Non-masked
        path_full = f'{dir_registry["reconstruction_dir"]}/{model_name}_full.png'
        Image.fromarray(np.array(reconstructed_signal).astype(np.uint8)).save(path_full)
        
        ## Masked
        path_mask = f'{dir_registry["reconstruction_dir"]}/{model_name}_masked.png'
        reconstructed_signal = jnp.concat((reconstructed_signal, jnp.expand_dims(sampler.alpha, axis=-1)*255), axis=-1)
        Image.fromarray(np.array(reconstructed_signal).astype(np.uint8)).save(path_mask)

        return path_full, path_mask
