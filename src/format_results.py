import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from utils.registry import *
from utils.model.registry import *

if __name__ == "__main__":

    # Go over all data json
    folder = dir_registry["raw_analysis_data_dir"]
    for file in os.listdir(folder):

        # Initialize data cotainer for creating plots
        model_name, color, error, inference = [], [], [], []
        line_x, line_y, current_type = [], [], ""

        # Deserialize analysis data 
        with open(folder + "/" + file) as f:
            raw_data = json.load(f)

            # Mark loss difference epsilon value to be considered saturated
            # M2E below this line is nearly inachievable 
            plt.axhline(train_set["epsilon"], linestyle='--', c='black', label=r'$\epsilon$')

            # Extract analysis criteria
            for key, value in raw_data.items():
                model_type = key.split('_')[0]
                model_name.append(key)
                error.append(value[a_registry['keys']['error']])
                color.append(a_registry['mcol'][model_type])
                inference.append(value[a_registry['keys']['inference']])

                # Create plot and save
                ## Draw line between models
                if current_type != model_type: 
                    if current_type != '' : plt.plot(line_x, line_y, color=a_registry['mcol'][current_type], label=current_type)
                    line_x, line_y, current_type = [inference[-1]], [error[-1]], model_type
                else: 
                    line_y.append(error[-1])
                    line_x.append(inference[-1])
            plt.plot(line_x, line_y, color=a_registry['mcol'][current_type], label=current_type)

            ## Plot model points, annotate and label axis
            plt.scatter(
                inference,
                error,
                s=90,
                c=color)
            plt.ylabel("M2E")
            plt.xlabel(f"Inference speed (μs) per single query\n Mean of {a_registry['inf_reps']} repetitions")
            for i in range(len(model_name)):
                plt.annotate(
                    model_name[i].split('_')[1].replace("0", ""),
                    (inference[i], error[i]),
                    ha='center',
                    va='center',
                    c='white',
                    fontweight='bold',
                    fontsize=6)
            plt.legend()
            
            # Save plot with timestamp to avoid overide
            dir = dir_registry["analysis_plot_dir"]
            if not os.path.isdir(dir):
                os.makedirs(dir)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            plt.tight_layout()
            plt.savefig(f"{ dir + "/" + timestamp}.png", dpi=300) 
