from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# Base class for sampling training data
# Relevant when changing training data format
class DataSampler(ABC):

    @property
    @abstractmethod
    def x_dim(self):
        """The signal input dimensions."""
        pass

    @property
    @abstractmethod
    def y_dim(self):
        """The signal output dimensions."""
        pass
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def check_signal(self):
        pass

    @staticmethod
    def batch_generator(x, batch_size):
        batches =[]
        for i in range(0, x.shape[0], batch_size):
            batches.append(x[i:i + batch_size])
            
        # Trim tail of x that does not fit with batchsize
        return jnp.stack(batches[0:-1])

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
        self._x_dim = 2  
        self._y_dim = 3 

    @property
    def x_dim(self): return self._x_dim

    @property
    def y_dim(self): return self._y_dim

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
            The number of samples.
        key :
            Jax random key used for sampling

        Returns
        ----------
        samples :
            The samples [coordinate, color]
            All values are in range 0.0 - 1.0
        """
        p_x, p_y = jnp.meshgrid(jnp.arange(self.img.shape[0]),jnp.arange(self.img.shape[1]), indexing='ij')
        p_x = jnp.stack((p_x, p_y),axis=-1)[self.alpha]
        x = p_x[jax.random.choice(jax.random.split(key)[1], p_x.shape[0], (n_samples,), replace=False)]
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

        Returns
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
    
    def inference_sample(self):
        """
        Extract all pixel coordinates (x) to fully infere a model with the image used for this sampler.

        Returns
        ----------
        coordinates :
            All image coordinates from the sampled image [coordinate], shape [width, height, XY coordinate]
            All values are in range 0.0 - 1.0
        """
        p_x, p_y = jnp.meshgrid(jnp.arange(self.img.shape[0]),jnp.arange(self.img.shape[1]), indexing='ij')
        x = jnp.stack((p_x, p_y),axis=-1)
        x = x.at[...,0].divide(self.img.shape[0])
        x = x.at[...,1].divide(self.img.shape[1])
        return x
    
    def inference_sample_solid(self):
        """
        Extract all non-transparent, solid pixel coordinates (x) to fully infere 
        a model with the solid regions of the image used for this sampler.

        Returns
        ----------
        (coordinates, RGB) :
            Image coordinates for solid image region, shape [width, height, XY coordinate] 
            and their channels, shape [width, height, RGB] 
            All values are in range 0.0 - 1.0
        """
        p_x, p_y = jnp.meshgrid(jnp.arange(self.img.shape[0]),jnp.arange(self.img.shape[1]), indexing='ij')
        x = jnp.stack((p_x, p_y),axis=-1)[self.alpha]
        y = self.rgb[x[...,0], x[...,1]]
        x = x.at[...,0].divide(self.img.shape[0])
        x = x.at[...,1].divide(self.img.shape[1])
        return x, y

         