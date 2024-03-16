import os
import pickle
import numpy as np
from config import *
from torch.utils.tensorboard import SummaryWriter

dir_path = "metrics_data/train/fixed/optimal/"
files = os.listdir(dir_path)

for file in files:
    print(file)
    file_name = file.split('_')[1]
    with open(f"{dir_path}/{file}", "rb") as f:
        train = pickle.load(f)
   
    def safe_mean(arr):
        arr = np.array(arr)
        arr[np.isinf(arr)] = np.nan  # Replace infinities with NaN
        return np.nanmean(arr)  # Compute mean ignoring NaN

    random_train_avg = {key: safe_mean(value) for key, value in train.items()}

    log_dir = f"runs/train/fixed/"
    writer = SummaryWriter(log_dir)

    for epoch in range(120):
        writer.add_scalar('total adjusted reward (per pickup distance)', random_train_avg['total_adjusted_reward'], epoch)
        writer.add_scalar('total reward', random_train_avg['total_reward'], epoch)
        writer.add_scalar('total orders', random_train_avg['total_request_num'], epoch)
        writer.add_scalar('matched orders', random_train_avg['matched_request_num'], epoch)
        writer.add_scalar('matched request ratio', random_train_avg['matched_request_ratio'], epoch)
        writer.add_scalar('matched occupancy rate', random_train_avg['occupancy_rate'], epoch)
        writer.add_scalar('matched occupancy rate - no pickup', random_train_avg['occupancy_rate_no_pickup'], epoch)
        writer.add_scalar('pickup time', random_train_avg['pickup_time'], epoch)
        writer.add_scalar('waiting time', random_train_avg['waiting_time'], epoch)

    writer.close()