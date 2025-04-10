# Model specific directories
dir_registry = {
    "reconstruction_dir" : "results/reconstruction", # Folder to store full reconstructed signal for brief validation of model
    "model_params_dir" : "results/models", # Folder to store the serialized model parameters for each trained model
    "raw_analysis_data_dir" : "results/raw", # Folder to store serialized model analysis results
    "analysis_plot_dir" : "results/plot" # Folder to store plot with formatted analysis data
}

# Model analysis hyperparameters
a_registry = {
    # Inference repitions for average time measure
    'inf_reps' : 200,
    # Model analysis criteria
    'keys' : {
        "inference" : "avg_inference",
        "error" : " m2e"
    },
    # Plot colors for model types
    'mcol' : {
        'mlp' : 'blue',
        'moe' : 'magenta',
    }
}