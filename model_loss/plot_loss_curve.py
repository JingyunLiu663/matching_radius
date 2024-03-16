import matplotlib.pyplot as plt

# Initialize lists to hold parsed data
epochs = []
average_losses = []

# Open the log file and parse the data
with open('2024-03-16 14:31:30.998151_dqn_loss.log' , 'r') as log_file:
    for line in log_file:
        # Remove newline character and any leading/trailing whitespace
        line = line.strip()
        # Split the line by comma and then by colon
        epoch_str, loss_str = line.split(", ")
        # Extract the epoch number
        _, epoch_num = epoch_str.split(":")
        # Extract the average loss
        _, avg_loss = loss_str.split(":")
        # Append to the lists
        epochs.append(int(epoch_num))
        average_losses.append(float(avg_loss))

# Plotting
plt.plot(epochs, average_losses, marker='o')
plt.title('Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.grid(True)
plt.savefig("loss_curve.png")