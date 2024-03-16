import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn's style
sns.set_theme()

# Load the record dictionary
with open('fixed1k_radius_train.pkl', 'rb') as f:
    record_dict = pickle.load(f)

# Make sure the visualization directory exists
os.makedirs('./visualization/fixed1k_train', exist_ok=True)

# Plotting the records
for record_name, record_values in record_dict.items():
    plt.figure(figsize=(10, 5))  # Adjust as necessary
    sns.lineplot(x=range(len(record_values)), y=record_values)
    plt.xlabel('Epoch')
    plt.ylabel(record_name)
    plt.title(record_name)
    plt.savefig(f'./visualization/fixed1k_train/{record_name}.png')
    plt.close()