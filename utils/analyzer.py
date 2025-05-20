import time
from utils.model.BaseModel import *
from utils.DataSampler import *
from flax import struct

class Analyzer():
    """
    <summary>
        Handels inference analysis of a model.
    </summary>
    """

    def __init__(self, sampler:DataSampler):
        """
        Instantiates analyzer for a datasampler. All evaluations measure only inferrence.

        Args
        ----------
        sampler :
            The data sampler used for training the model. Going to be used for inferrence analysis
        """
        self.sampler = sampler

    def eval_inference_speed_IMG(self, n:int , model_type:BaseModel, p:BaseParams):    
        """
        Measures the average inference time for the sampler signal over n repetitions in miliseconds.

        Args
        ----------
        n :
            Number of inference repetitions of the full signal
        model_type :
            The model type
        p :
            The model parameters
        sampler :
            The sampler used for training the model

        Returns
        ----------
        inference average :
            The average inference time for n repetitions. 
            Time is only taken for batched forward call. 
            Huge forward calls might require mini-batching.
        """
        # Prepare Image for analysis and flatten into 
        x, _ = self.sampler.inference_sample_solid()
        x_flat = x.reshape(-1, x.shape[-1])
        batch_size = 128
        x_batched = DataSampler.batch_generator(x, batch_size)

        # Timed inference
        ## Warm up JIT and trace, compile and cache 
        model_type.forward(p, x_flat).block_until_ready()
        time0 = time.time()
        for i in range(n):
            out = jax.vmap(lambda x: model_type.forward(p, x))(x_batched)
            out.block_until_ready()

        ## convert to microseconds (1s = 1000000 μs)
        inf_timing = (time.time() - time0) / (jnp.size(x_batched, axis=0) * batch_size) / n * 1000000
        return inf_timing

    def eval_accuracy_IMG(self, model_type:BaseModel, p:BaseParams):
        """
        Computes the mean squared error for the reconstruction
        of the non-transparent, solid image signal.

        Args
        ----------
        model_type :
            The model type
        p :
            The model parameters
        sampler :
            The sampler used for training the model

        Returns
        ----------
        m2e :
            Mean squared error
        """

        # Use loss function to compute accuracy 
        # Requires all loss functions to be evaluated the same to allow for comparison
        x, y = self.sampler.inference_sample_solid()
        m2e = model_type.loss(p,x,y)

        return m2e