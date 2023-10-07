import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import *
from utilities import *

"""
This script is used for RL to learn the optimal matching radius
DQN with replay buffer and fixed target is implemented
Attention:
    State may be stored as a customized State object in other scripts, 
    but within this RL agent construct script, State is maintained in tuple format for memory efficiency consideration
"""


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
        actions = self.out(x)
        return actions


class DqnAgent:
    """
        TODO: action space [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        The is an agent for DQN learning for dispatch problem:
        Agent: individual driver, each driver is taken as an agent, but all the agents share the same DRL parameters
        Reward: r / radius (Immediate trip fare regularized by radius)
        State: tuple (time_slice, grid_id)
        Action: matching radius applied (km)
    """

    def __init__(self, action_space: list, num_layers: int, layers_dimension_list: list, lr=0.005, gamma=0.9, epsilon=0.9, eps_min=0.01,
                 eps_dec=0.997, target_replace_iter=100, batch_size=8, mem_size=2000):
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
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.transition_count = 0  # number of transition (s_t, a_t, r_t, s_{t + 1}) recorded

        # one network as the network to be evaluated, the other as the fixed target
        self.eval_net = DqnNetwork(input_dims=self.input_dims, num_layers=self.num_layers,
                                   layers_dimension_list=self.layers_dimension_list,
                                   n_actions=self.num_actions, lr=self.lr)
        self.target_net = DqnNetwork(input_dims=self.input_dims, num_layers=self.num_layers,
                                     layers_dimension_list=self.layers_dimension_list,
                                     n_actions=self.num_actions, lr=self.lr)

        # to plot the loss curve·
        self.loss_values = []

    def store_transition(self, states, actions, rewards, next_states):
        '''
        Store one transition record (s_t, a_t, r_t, s_{t + 1}) in the replay buffer;
        The data is stored in 5 numpy arrays respectively.
        :param states: tuple(time_slice, grid_id)
        :param actions: np.int32 -> action index, shall be used with action_space to get the true matching radius value
        :param rewards: np.float32
        :param next_states: tuple(time_slice, grid_id)
        '''
        n_transitions = len(states)

        # Calculate indices to store the new transitions
        start_index = self.transition_count % self.mem_size
        end_index = start_index + n_transitions

        if end_index <= self.mem_size:  # If new data fits within the buffer without wrapping around
            self.state_memory[start_index:end_index] = states
            self.new_state_memory[start_index:end_index] = next_states
            self.reward_memory[start_index:end_index] = rewards
            self.action_memory[start_index:end_index] = actions

        else:  # If new data exceeds the buffer's end and needs to wrap around
            # First, fill until the buffer's end
            n_to_end = self.mem_size - start_index
            self.state_memory[start_index:] = states[:n_to_end]
            self.new_state_memory[start_index:] = next_states[:n_to_end]
            self.reward_memory[start_index:] = rewards[:n_to_end]
            self.action_memory[start_index:] = actions[:n_to_end]

            # Then, wrap around and overwrite from the beginning
            n_remaining = n_transitions - n_to_end
            self.state_memory[:n_remaining] = states[n_to_end:]
            self.new_state_memory[:n_remaining] = next_states[n_to_end:]
            self.reward_memory[:n_remaining] = rewards[n_to_end:]
            self.action_memory[:n_remaining] = actions[n_to_end:]

        self.transition_count += n_transitions
        self.degug_log()

    def choose_action(self, observation: tuple):
        """
        Choose action based on epsilon-greedy algorithm
        :param observation: tuple(time_slice, grid_id)
        :return: action index (to be used with self.action_space later)
        """
        if np.random.random() > self.epsilon:
            # Convert observation (tuple) into a tensor with shape (1, 2)
            state_tensor = torch.tensor([observation], dtype=torch.float32)
            # Greedy action selection
            with torch.no_grad():
                actions = self.eval_net(state_tensor)
                action_index = torch.argmax(actions).item()
        else:
            # randomize
            action_index = np.random.randint(self.num_actions)
        return action_index

    def learn(self):
        # TODO: only start learning when one batch is filled? If not, keep collecting transition data
        if self.transition_count < self.batch_size:
            return

        # update the target network parameter
        if self.transition_count % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # max_mem: get the number of available transition records (to sample from)
        max_mem = min(self.transition_count, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)  # TODO: sample without replacement?

        # convert numpy array to tensor
        state_batch = torch.tensor(self.state_memory[batch], dtype=torch.float32)
        reward_batch = torch.tensor(self.reward_memory[batch], dtype=torch.float32)
        new_state_batch = torch.tensor(self.new_state_memory[batch], dtype=torch.float32)
        action_batch = torch.tensor(self.action_memory[batch].squeeze(), dtype=torch.int64)

        # RL learn by batch
        q_eval = self.eval_net(state_batch)[np.arange(self.batch_size), action_batch]
        # Make sure q_eval's first dimension is always equal to batch_size, otherwise the above code will cause error
        q_next = self.target_net(new_state_batch)  # target network shall be applied here
        # Side notes:
        #   q_next is a 2-dimensional tensor with shape: (batch_size, num_actions)
        #   torch.max() returns a tuple:
        #     The first element ([0]) is the actual maximum values (in our case, the Q-values).
        #     The second element ([1]) is the indices of these max values.
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        # calculate loss and do the back-propagation
        loss = self.eval_net.loss(q_target, q_eval)
        # to plot the loss curve
        self.loss_values.append(loss.item())
        self.eval_net.optimizer.zero_grad()
        loss.backward()
        self.eval_net.optimizer.step()

        # update epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


    def save_parameters(self, path: str):
        """
        Save model parameters.
        
        Args:
        - path (str): The path to save the model parameters.
        """
        torch.save(self.eval_net.state_dict(), path)

    def load_parameters(self, path: str):
        """
        Load model parameters.
        
        Args:
        - path (str): The path to load the model parameters from.
        """
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))