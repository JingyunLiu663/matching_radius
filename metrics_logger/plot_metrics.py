import pandas as pd
import matplotlib.pyplot as plt
import re

log_file_path = '2024-03-16 13:14:17.309596_fixed_r1.0.log'

# Initialize a dictionary to store the parsed data
data = {
    'epoch': [],
    'total_reward': [],
    'matching_rate': [],
    'total_request_num': [],
    'matched_request_num': [],
    'occupancy_rate': [],
    'occupancy_rate_no_pickup': [],
    'wait_time': [],
    'pickup_time': []
}

# Read the log file and parse the data
with open(log_file_path, 'r') as file:
    for line in file:
        # Use regular expression to extract values
        pattern = r'epoch: (\d+) == total_reward:(.*),matching_rate:(.*),total_request_num:(.*),matched_request_num:(.*),occupancy_rate:(.*),occupancy_rate_no_pickup:(.*),wait_time:(.*),pickup_time:(.*)'
        match = re.search(pattern, line)
        if match:
            data['epoch'].append(int(match.group(1)))
            data['total_reward'].append(float(match.group(2)))
            data['matching_rate'].append(float(match.group(3)))
            data['total_request_num'].append(float(match.group(4)))
            data['matched_request_num'].append(float(match.group(5)))
            data['occupancy_rate'].append(float(match.group(6)))
            data['occupancy_rate_no_pickup'].append(float(match.group(7)))
            data['wait_time'].append(float(match.group(8)))
            data['pickup_time'].append(float(match.group(9)))

# Convert the parsed data into a Pandas DataFrame
df = pd.DataFrame(data)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot total reward
axs[0, 0].plot(df['epoch'], df['total_reward'], marker='o')
axs[0, 0].set_title('Total Reward per Epoch')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Total Reward')

# Plot matching rate
axs[0, 1].plot(df['epoch'], df['matching_rate'], marker='o', color='orange')
axs[0, 1].set_title('Matching Rate per Epoch')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Matching Rate')

# Plot occupancy rate
axs[1, 0].plot(df['epoch'], df['occupancy_rate'], marker='o', label='With Pickup')
axs[1, 0].plot(df['epoch'], df['occupancy_rate_no_pickup'], marker='o', linestyle='--', label='No Pickup')
axs[1, 0].set_title('Occupancy Rate per Epoch')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Occupancy Rate')
axs[1, 0].legend()

# Plot wait time
axs[1, 1].plot(df['epoch'], df['wait_time'], marker='o', color='green')
axs[1, 1].set_title('Wait Time per Epoch')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Wait Time')

# Save the plot
plt.tight_layout()
plt.savefig("fixed_r1.0_metrics.png")