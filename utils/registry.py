# Model specific directories
dir_registry = {
    "reconstruction_dir" : "results/reconstruction", # Folder to store full reconstructed signal for brief validation of model
    "model_params_dir" : "models", # Folder to store the serialized model parameters for each trained model
    "raw_analysis_data_dir" : "results/raw", # Folder to store serialized model analysis results
    "analysis_plot_dir" : "results/plot" # Folder to store plot with formatted analysis data
}

# Model analysis criteria
analysis_keys = {
    "inference" : "avg_inference",
    "error" : " m2e"
}