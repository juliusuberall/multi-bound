import argparse
import jax
import os
from utils.model import *
from utils.registry import * 
from utils.sampler import RGBAImageSampler

if __name__ == "__main__":

    # Initialize random key
    key = jax.random.PRNGKey(29)

    # Parse invoking arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data object to fit")
    args = vars(parser.parse_args())

    # Load training signal sampler
    sampler = RGBAImageSampler(args['data'])
    sampler.check_signal(key)

    # Load model parameters and analyze
    for param_file in os.listdir("models"):
        print(f"Analyzing {param_file}")

        ## Identify model type
        model_type = model_registry[param_file.split('_')[0]]

        ## Deserialize JSON and retrieve model parameters
        p = model_type.deserialize(model_dir["model_params_dir"] + "/" + param_file)

        ## Inference of full training signal to validate
        model_type.full_signal_inference_IMG(p, sampler,param_file.split('.')[0])

        ## Accuracy measure