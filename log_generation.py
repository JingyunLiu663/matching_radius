import numpy as np
import pickle
from datetime import datetime
import os
import logging
from torch.utils.tensorboard import SummaryWriter
'''
Script Overview:
- `ModelTracker`: Saves model parameters to 'models' directory.
- `ActionTracker`: Logs action distributions per timestep at date and epoch level. Saved to `actions_collection` directory.
- `EpochPerformanceTracker`: Records performance metrics to `metrics_data` directory and feeds them to TensorBoard for live monitoring. 
'''
logging.basicConfig(level=logging.INFO)

class ModelTracker:
    '''
    Saves model parameters to 'models' directory
    '''
    def __init__(self):
        pass

    def save_model(self, agent, rl_agent, epoch, adjust_reward=0):
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        parameter_path = os.path.join(models_dir, f"{rl_agent}_epoch{epoch}_{adjust_reward}_model.pth")
        agent.save_parameters(parameter_path)

class ActionTracker:
    '''
        An epoch consists of a list of dates.
        A list of action distributions in regard to time step within each date.
        Demo:
            actions_dict[epoch][date] = [
                {0: 52, 1: 64, ...}, # at time step 0
                {0: 36, 1: 49, ...}, # at time step 1
                ...
            ]
        Saved to `actions_collection` directory.
    '''
    def __init__(self):
        # Initialize a dictionary to hold action distribution dictionaries for each epoch, date, and timestep
        self.actions_dict = {} # key is the epoch
        self.action_counts_per_timestep = []
        self.time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')

    def init_new_epoch(self, epoch):
        if epoch in self.actions_dict:
            print(f"Warning: epoch {epoch} has already been initialized. Overwriting data.")
        self.actions_dict[epoch] = {}

    def init_new_date(self, epoch, date):
        if epoch not in self.actions_dict:
            print(f"Warning: epoch {epoch} has not been initialized. Initialize epoch before date.")
            return

        if date in self.actions_dict[epoch]:
            print(f"Warning: date {date} in epoch {epoch} has already been initialized. Overwriting data.")
        self.actions_dict[epoch][date] = []

    def insert_time_step(self, args, action_indices):
        '''
        action_indices is a list
        '''
        # Create a dictionary to hold action counts for this time step
        action_counts = {i: action_indices.count(i) for i in range(len(args.action_space))}
        self.action_counts_per_timestep.append(action_counts)

    def end_time_step(self, epoch, date):
        if epoch not in self.actions_dict:
            print(f"Error: epoch {epoch} has not been initialized. Initialize epoch before ending time step.")
            return

        if date not in self.actions_dict[epoch]:
            print(f"Error: date {date} in epoch {epoch} has not been initialized. Initialize date before ending time step.")
            return

        self.actions_dict[epoch][date] = self.action_counts_per_timestep # assign the actions distribution on that date
        self.action_counts_per_timestep = [] # reset the time step tracker to an empty list

    def save_actions_distribution(self, arg):
        actions_dir = os.path.join("actions_collection", arg.experiment_mode, arg.radius_method)
        os.makedirs(actions_dir, exist_ok=True)
        actions_file = os.path.join(actions_dir, f'{arg.rl_agent}_{arg.adjust_reward}_{self.time_stamp}.pkl')
        with open(actions_file, 'wb') as f:
            pickle.dump(self.actions_dict, f)

class EpochPerformanceTracker:
    '''
    The performance is tracked at epoch level
    - Serialized results under `metrics_data` directory
    - TensorBoard for live monitoring
    '''
    def __init__(self, experiment_mode, radius_method="rl", date=None, radius=None):
        # Define the columns for recording metrics
        self.column_list = [
            'epoch_average_loss', 'total_adjusted_reward', 'total_reward',
            'total_request_num', 'matched_request_num',
            'matched_request_ratio',
            'waiting_time', 'pickup_time',
            'occupancy_rate', 'occupancy_rate_no_pickup'
        ]
        
        # Initialize record dictionary with empty lists
        self.record_dict = {col: [] for col in self.column_list}
        self.time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')
        
        # Set experiment mode and reset metrics
        self.experiment_mode = experiment_mode
        self.radius_method = radius_method
        self._reset_metrics()

        if radius_method != "rl":
            log_dir = os.path.join("runs", experiment_mode, radius_method, f"{date}_{radius}_{self.time_stamp}.pkl")
            self.baseline_writer = SummaryWriter(log_dir)

    def reset(self):
        # Reset all metrics
        self._reset_metrics()

    def _reset_metrics(self):
        # Reset metrics that are always tracked
        self.episode_time = []
        self.total_reward = []
        self.total_adjusted_reward = []
        self.total_request_num = []
        self.matched_requests_num = []
        self.occupancy_rate = []
        self.occupancy_rate_no_pickup = []
        self.pickup_time = []
        self.waiting_time = []
        
        # Reset the epoch loss if in train mode
        if self.experiment_mode == "train":
            self.epoch_loss = []

    def add_loss_per_learning_step(self, loss):
        self.epoch_loss.append(loss)

    def add_within_epoch(self, reward, adjust_reward, request_num, matched_request_num, 
                 occupancy_rate, occupancy_rate_no_pickup, pickup_time, wait_time, time):
        self.episode_time = time
        self.total_reward.append(reward)
        self.total_adjusted_reward.append(adjust_reward)
        self.total_request_num.append(request_num)
        self.matched_requests_num.append(matched_request_num)
        self.occupancy_rate.append(occupancy_rate)
        self.occupancy_rate_no_pickup.append(occupancy_rate_no_pickup)
        self.pickup_time.append(pickup_time / matched_request_num) # average pickup time per order
        self.waiting_time.append(wait_time / matched_request_num) # average wait time per order

    def _add_scalar_to_tensorboard(self, writer, metric_name, value, epoch):
        writer.add_scalar(metric_name, value, epoch)

    def add_to_tensorboard(self, epoch, agent=None, use_rl=True):
        # Choose the appropriate writer based on whether we're using RL or the baseline
        if use_rl:
            writer = agent.train_writer if self.experiment_mode == "train" else agent.test_writer
        else:
            writer = self.baseline_writer

        metrics = {
            'running time': np.mean(self.episode_time),
            'total adjusted reward (per pickup distance)': sum(self.total_adjusted_reward),
            'total reward': sum(self.total_reward),
            'total orders': sum(self.total_request_num),
            'matched orders': sum(self.matched_requests_num),
            'matched request ratio': sum(self.matched_requests_num) / sum(self.total_request_num) if self.total_request_num else 0,
            'matched occupancy rate': np.mean(self.occupancy_rate),
            'matched occupancy rate - no pickup': np.mean(self.occupancy_rate_no_pickup),
            'pickup time': np.mean(self.pickup_time),
            'waiting time': np.mean(self.waiting_time)
        }
        
        # Add epoch loss only for training mode and if using the RL agent
        if self.experiment_mode == "train" and use_rl:
            metrics['average loss'] = np.mean(self.epoch_loss)
        
        # Log the metrics
        for metric_name, value in metrics.items():
            self._add_scalar_to_tensorboard(writer, metric_name, value, epoch)

    def add_to_metrics_data(self):
        if self.experiment_mode == "train" and self.radius_method == "rl":
            self.record_dict['epoch_average_loss'].append(np.mean(self.epoch_loss))
        self.record_dict['total_adjusted_reward'].append(sum(self.total_adjusted_reward))
        self.record_dict['total_reward'].append(sum(self.total_reward))
        self.record_dict['total_request_num'].append(sum(self.total_request_num))
        self.record_dict['matched_request_num'].append(sum(self.matched_requests_num))
        self.record_dict['matched_request_ratio'].append(sum(self.matched_requests_num) / sum(self.total_request_num))
        self.record_dict['occupancy_rate'].append(np.mean(self.occupancy_rate))
        self.record_dict['occupancy_rate_no_pickup'].append(np.mean(self.occupancy_rate_no_pickup))
        self.record_dict['pickup_time'].append(np.mean(self.pickup_time))
        self.record_dict['waiting_time'].append(np.mean(self.waiting_time))

    def save(self, experiment_mode,  radius_method="rl", adjust_reward=0, rl_agent=None, radius=None, date=None):
        '''
        rl_agent: dqn, ddqn, dueling_dqn
        adjust_reward: 0/1
        experiment_mode: "train"/"test"
        radius and date is for fixed mode
        '''
        # output directories
        metrics_dir = os.path.join("metrics_data", experiment_mode, radius_method)
        # create directories if they do not exist
        os.makedirs(metrics_dir, exist_ok=True)
        # output files
        if radius_method == "fixed":
            metrics_file = os.path.join(metrics_dir, f'{date}_{radius}_{self.time_stamp}.pkl')
        elif radius_method == "rl":
            metrics_file = os.path.join(metrics_dir, f'{rl_agent}_{adjust_reward}_{self.time_stamp}.pkl')
        # serialize the record
        with open(metrics_file, 'wb') as f:
            pickle.dump(self.record_dict, f)
