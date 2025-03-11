from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# Base class for sampling training data
# Relevant when changing training data format
class DataSampler(ABC):
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def check_signal(self):
        pass

# Class to store and sample image to learn with network
class RGBAImageSampler(DataSampler):

    def __init__(self, img_path):
        """
        Define with an RGBA image the color signal to sample.

        Args
        ----------
        img_path :
            The location of the RGBA image.
        """
        self.path = img_path
        self.img = jnp.array(plt.imread(img_path))
        self.rgb = self.img[...,0:3]
        self.alpha = self.img[...,3].astype(bool)

    def check_signal(self, key):
        """
        Print some basic information about the image to ensure its formatted correct.
        
        Args
        ----------
        key :
            Jax random key used for sampling
        """
        print("=============================================")
        print(f"Checking image signal of {self.path}")
        print(f"Image intensity range: {jnp.min(self.img)} to {jnp.max(self.img)}")
        print(f"Example sample (Coordinate, RGB): {self.sample(1, key)}")
        print("=============================================")
        return None

    def sample(self, n_samples, key):
        """
        Sample non-transparent region of an RGBA image.

        Extract n samples at pixel coordinate (x) and with ground truth color (y)

        Args
        ----------
        n_samples :
            The input image as a NumPy or JAX array.
        key :
            Jax random key used for sampling

        Args
        ----------
        samples :
            The samples [coordinate, color]
        """
        p_x, p_y = jnp.meshgrid(jnp.arange(self.img.shape[0]),jnp.arange(self.img.shape[1]), indexing='ij')
        p_x = jnp.stack((p_x, p_y),axis=-1)[self.alpha]
        x = p_x[jax.random.choice(jax.random.split(key)[1], p_x.shape[0], (n_samples,))]
        y = self.rgb[x[...,0], x[...,1]]
        x = x.at[...,0].divide(self.img.shape[0])
        x = x.at[...,1].divide(self.img.shape[1])
        return x, y
    
    def sample_regions(self, n_samples, key, region_mask):
        """
        Sample non-transparent regions of the RGBA image. Group samples by 2 provided regions.

        Extract n samples at pixel coordinate (x) and with ground truth color (y)

        Args
        ----------
        n_samples :
            The input image as a NumPy or JAX array.
        key :
            Jax random key used for sampling
        region_mask :
            Boolean region mask of the image, dividing it into two regions.

        Args
        ----------
        samples :
            The samples [coordinate, color]
        """
        # Extract n non-masked pixel coordinates (x) and colors (y)
        p_x, p_y = jnp.meshgrid(jnp.arange(self.img.shape[0]),jnp.arange(self.img.shape[1]), indexing='ij')

        # Pick random samples in both regions equally
        # We use the alpha mask from the image combined with the region mask 
        # to sample the assigned region of the dolphin for each of the both models
        # -> There is no area weighting done for sampling the regions
        n_samples = int(n_samples / 2)
        p1_coords = jnp.stack((p_x, p_y),axis=-1)[self.alpha * region_mask]
        x1 = p1_coords[jax.random.choice(jax.random.split(key)[1], p1_coords.shape[0], (n_samples,))]
        p2_coords = jnp.stack((p_x, p_y),axis=-1)[self.alpha * ~region_mask]
        x2 = p2_coords[jax.random.choice(jax.random.split(key)[1], p2_coords.shape[0], (n_samples,))]
        x = jnp.stack((x1, x2), axis=0)
        
        # Normalize coordinates and extract color labels
        y = self.rgb[x[...,0], x[...,1]]
        x = x.at[...,0].divide(self.img.shape[0])
        x = x.at[...,1].divide(self.img.shape[1])

        return x, y