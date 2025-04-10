import time
from utils.model.BaseModel import *
from utils.sampler import *
from flax import struct

def eval_inference_speed_IMG(n:int , model_type:BaseModel, p:BaseParams, sampler:DataSampler):    
    """
    Measures the average inference time for the signal to fit over n repetitions.

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
        The average inference time for n repetitions
    """
    inf_timing = []
    for i in range(n):
        time0 = time.time()
        model_type.signal_inference_solid_IMG(p, sampler, model_type)
        time1 = time.time()
        inf_timing.append(time1 - time0)
    inf_timing = jnp.mean(jnp.array(inf_timing))
    return inf_timing

def eval_accuracy_IMG(model_type:BaseModel, p:struct.PyTreeNode, sampler:DataSampler):
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
    rec, y = model_type.signal_inference_solid_IMG(p, sampler, model_type)
    m2e = jnp.mean((rec - y)**2)
    return m2e