import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import *
from utilities import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

"""
This script is used for RL to learn the optimal matching radius
Double DQN with replay buffer and fixed target is implemented
Attention:
    State may be stored as a customized State object in other scripts, 
    but within this RL agent construct script, State is maintained in tuple format for memory efficiency consideration
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DqnNetwork(nn.Module):
    def __init__(self, input_dims: int, num_layers: int, layers_dimension_list: list, n_actions: int, lr: float):
        """
        :param input_dims: presumably 2 (time_slice, grid_id)
        :param num_layers: number of intermediate layers
        :param layers_dimension_list: a list indicating number of dimension of each layer
        :param n_actions: the action space
        :param lr: learning rate
        """
        super(DqnNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(nn.Linear(input_dims, layers_dimension_list[0]))

        # middle layers
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(layers_dimension_list[i - 1], layers_dimension_list[i]))

        # output layer
        self.out = nn.Linear(layers_dimension_list[-1], n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()


    def forward(self, state):
        """
        Use ReLu as the activation function
        :param state: A tensor representation of tuple (time_slice, grid_id)
        """
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        q_value = self.out(x)
        return q_value


class DDqnAgent:
    """
        TODO: action space [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        The is an agent for DQN learning for dispatch problem:
        Agent: individual driver, each driver is taken as an agent, but all the agents share the same DRL parameters
        Reward: r / radius (Immediate trip fare regularized by radius)
        State: tuple (time_slice, grid_id)
        Action: matching radius applied (km)
    """

    def __init__(self, action_space: list, num_layers: int, layers_dimension_list: list, lr=5e-4, gamma=0.99,
                 epsilon=1.0, eps_min=0.01, eps_dec=0.9978, target_replace_iter=2000, mode="train", adjust_reward=0):
        self.num_actions = len(action_space)
        self.num_layers = num_layers
        self.layers_dimension_list = layers_dimension_list
        self.input_dims = 2  # (grid_id, time_slice)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.target_replace_iter = target_replace_iter  # how often do we update the target network
        self.batch_size = BATCH_SIZE

        self.mode = mode

        self.eval_net_update_times = 0

        # one network as the network to be evaluated, the other as the fixed target
        self.eval_net = DqnNetwork(self.input_dims, self.num_layers, self.layers_dimension_list, self.num_actions,
                                   self.lr).to(device)
        self.target_net = DqnNetwork(self.input_dims, self.num_layers, self.layers_dimension_list, self.num_actions,
                                     self.lr).to(device)

        # to plot the loss curve
        self.loss = 0
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        if self.mode == "train":
            # Create a SummaryWriter object and specify the log directory
            train_log_dir = f"runs/train/experiment_ddqn_{adjust_reward}_{current_time}"
            self.train_writer = SummaryWriter(train_log_dir)
            hparam_dict = {'lr': self.lr, 'gamma': self.gamma, 'epsilon': self.epsilon, 'eps_min': self.eps_min, 'eps_dec': self.eps_dec, 'target_replace_iter': self.target_replace_iter}
            self.train_writer.add_hparams(hparam_dict, {})
            self.train_writer.close()
        elif self.mode == "test":
            test_log_dir = f"runs/test/experiment_ddqn_{adjust_reward}_{current_time}"
            self.test_writer = SummaryWriter(test_log_dir)

    def choose_action(self, states: np.array):
        """
        Choose action based on epsilon-greedy algorithm
        :param states: numpy array of shape n * 2
        :return: numpy array of action index
        """
        n = states.shape[0]
        # Convert all observations to a tensor
        state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        # Compute Q-values for all states in one forward pass
        with torch.no_grad():
            q_values = self.eval_net(state_tensor)
        # Default action selection is greedy
        action_indices = torch.argmax(q_values, dim=1).cpu().numpy()
        if self.mode == "train":
            # Identify agents that should explorde
            explorers = np.random.random(n) < self.epsilon
            # Generate random actions for explorers
            action_indices[explorers] = np.random.randint(self.num_actions, size=np.sum(explorers))
        
        return action_indices

    def learn(self, states, action_indices, rewards, next_states):

        # update the target network parameter
        if self.eval_net_update_times % self.target_replace_iter == 0:
            self.update += 1
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # convert numpy array to tensor
        state_batch = torch.tensor(states, dtype=torch.float32).to(device)
        action_batch = torch.tensor(action_indices, dtype=torch.int64).to(device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(device)
        new_state_batch = torch.tensor(next_states, dtype=torch.float32).to(device)

        # RL learn by batch
        q_eval = self.eval_net(state_batch)[np.arange(BATCH_SIZE), action_batch]
        # Make sure q_eval's first dimension is always equal to batch_size, otherwise the above code will cause error

        # In Double DQN, we use the action selected by the evaluation network
        _, next_action_batch = self.eval_net(new_state_batch).max(1)  
        
        # Then we use this action to compute the Q-value from the target network for the next state
        q_next = self.target_net(new_state_batch)[np.arange(BATCH_SIZE), next_action_batch]
        q_target = (reward_batch + self.gamma * q_next.detach())

        # calculate loss and do the back-propagation
        loss = self.eval_net.loss(q_target, q_eval)

        # to plot the loss curve
        self.loss = loss.item()

        self.eval_net.optimizer.zero_grad()
        loss.backward()
        self.eval_net.optimizer.step()
        
        # update epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        self.eval_net_update_times += 1

    def save_parameters(self, path: str):
        """
        Save model parameters.
        
        Args:
        - path (str): The path to save the model parameters.
        """
        torch.save({
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict()
        }, path)


    def load_parameters(self, path: str):
        """
        Load model parameters.
        
        Args:
        - path (str): The path to load the model parameters from.
        """
        checkpoint = torch.load(path)
        self.eval_net.load_state_dict(checkpoint['eval_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.eval_net = self.eval_net.to(device)
        self.target_net = self.target_net.to(device)