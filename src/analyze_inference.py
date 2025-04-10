import argparse
import jax
import os
import json
from utils.analyzer import *
from utils.model.BaseModel import *
from utils.registry import * 
from utils.sampler import RGBAImageSampler
from utils.model.registry import *

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
    results = {}
    for param_file in sorted(os.listdir(dir_registry['model_params_dir'])):

        ## House keeping
        print(f"Analyzing {param_file}")
        model_results = {}

        ## Identify model type
        model_type = model_registry[param_file.split('_')[0]]

        ## Deserialize JSON and retrieve model parameters
        p = model_type.deserialize(dir_registry["model_params_dir"] + "/" + param_file)

        ## Inference and save of full training signal to visually validate model
        model_name = param_file.split('.')[0]
        model_type.save_full_signal_inference_IMG(p, sampler, model_name, model_type)

        ## M2E and inference-performance measure
        ### Inference-performance
        avg_inf = eval_inference_speed_IMG(a_registry['inf_reps'], model_type, p, sampler)
        model_results[a_registry['keys']["inference"]] = float(avg_inf)

        ### M2E 
        m2e = eval_accuracy_IMG(model_type, p, sampler)
        model_results[a_registry['keys']["error"]] = float(m2e)

        ## Store all analysis results for this model
        results[f'{model_name}'] = model_results
    
    # Serialize and save results
    ## Create results directory and save result data
    dir = dir_registry["raw_analysis_data_dir"]
    if not os.path.isdir(dir):
        os.makedirs(dir)
    with open(f"{dir}/data.json", "w") as f:
        json.dump(results, f)

        