import matplotlib as mp
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from utils.registry import *

# Simple stylesheet for different architectures
model_colors = {
    'mlp' : 'blue'
}

if __name__ == "__main__":

    # Go over all data json
    folder = dir_registry["raw_analysis_data_dir"]
    for file in os.listdir(folder):

        # Initialize data cotainer for creating plots
        model_name, color, error, inference = [], [], [], []

        # Deserialize analysis data 
        with open(folder + "/" + file) as f:
            raw_data = json.load(f)

            # Extract analysis criteria
            for key, value in raw_data.items():
                model_name.append(key)
                error.append(value[analysis_keys['error']])
                color.append(model_colors[key.split('_')[0]])

                ## convert to miliseconds (1s = 1000ms)
                inference.append(value[analysis_keys['inference']] * 1000)

            # Create plot and save
            plt.plot(inference, error)
            plt.scatter(inference, error, c=color)
            plt.ylabel("M2E")
            plt.xlabel("Inference speed (ms)\n Mean of 100 iterations full image prediction")
            for i in range(len(model_name)):
                plt.annotate(model_name[i], (inference[i], error[i]))
            
            # Save plot with timestamp to avoid overide
            dir = dir_registry["analysis_plot_dir"]
            if not os.path.isdir(dir):
                os.makedirs(dir)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            plt.tight_layout()
            plt.savefig(f"{ dir + "/" + timestamp}.png") 
