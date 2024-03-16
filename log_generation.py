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