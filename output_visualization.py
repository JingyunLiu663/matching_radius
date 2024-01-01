import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. load pickled files and calculate the average
def load_pickle_get_average(directory_path):
    # use a dictionary to store the average metrics per radius
    avg_metrics = {}
    date = os.listdir(directory_path)[0].split('_')[0]
    for file in os.listdir(directory_path):
        # get the radius
        radius = file.split('_')[1]
        # load the file
        with open(os.path.join(directory_path, file), "rb") as f:
            pickled_dict = pickle.load(f)
        for metric, ls in pickled_dict.items():
            if metric == "epoch_average_loss": continue
            if metric not in avg_metrics:
                avg_metrics[metric] = [(radius, np.mean(ls))]
            else:
                avg_metrics[metric].append((radius, np.mean(ls)))
    # sort the tuples by radius
    for key in avg_metrics.keys():
        avg_metrics[key].sort(key=lambda x: x[0])
    return avg_metrics, date

# Apply the default Seaborn theme styling to matplotlib
sns.set_theme()

# 2. Plot the line curves in separate figures 
def plot_data(data, date, folder_path):
    for key, values in data.items():
        # Handle empty data or all NaN values
        if len(values) == 0:
            print(f"No data available for key {key}. Skipping plot.")
            continue

        # Start a new figure
        plt.figure(figsize=(10, 8))
        # Create a DataFrame for Seaborn
        df = pd.DataFrame(values, columns=['Radius', 'Average Value'])
        # Use Seaborn to plot the data
        sns.lineplot(x='Radius', y='Average Value', data=df, marker='o')
        plt.xlabel('Radius')
        plt.ylabel('Average Value')
        plt.title(f'{key}')
        
        # Save the figure to the specified directory
        output_filename = os.path.join(folder_path, f'{date}_{key}.png')
        plt.savefig(output_filename)
        
        # Clear the figure after saving to avoid overlap
        plt.clf()


directory_path = "/home/jingyunliu/order-dispatch-radius/metrics_data/train/fixed"
avg_metrics, date = load_pickle_get_average(directory_path)
print(avg_metrics)
output_path ="/home/jingyunliu/order-dispatch-radius/output_visualization"
plot_data(avg_metrics, date, output_path)