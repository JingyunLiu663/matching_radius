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
    def __init__(self, lr: float, input_dims: int, fc1_dims: int, fc2_dims: int, n_actions: int):
        """
        TODO: which optimizer to use? Designed as a tunable configuration?
        :param lr: learning rate
        :param input_dims: input dimension -> state dimension in the RL model
        :param fc1_dims: fully-connected layer1 (hidden layer)
        :param fc2_dims: fully-connected layer2 (hidden layer)
        :param n_actions: # of actions
        """
        super(DqnNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        """
        Use ReLu as the activation function
        :param state: A tensor representation of tuple (time_slice, grid_id)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
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

    def __init__(self, **params):
        self.input_dims = params["input_dims"]
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon"]
        self.eps_min = params["eps_min"]
        self.eps_dec = params["eps_dec"]
        self.lr = params["lr"]
        self.num_actions = len(env_params['radius_action_space'])  # TODO
        self.target_replace_iter = params["target_replace_iter"]  # how often do we update the target network
        self.mem_size = params["max_mem_size"]
        self.batch_size = params["batch_size"]
        self.transition_count = 0  # number of transition (s_t, a_t, r_t, s_{t + 1}) recorded

        # one network as the network to be evaluated, the other as the fixed target
        # TODO: make DQN backbone part of the configuration file (so that it can be easily customized and tuned)
        self.eval_net = DqnNetwork(self.lr, input_dims=self.input_dims, fc1_dims=256, fc2_dims=256,
                                   n_actions=self.num_actions)
        self.target_net = DqnNetwork(self.lr, input_dims=self.input_dims, fc1_dims=256, fc2_dims=256,
                                     n_actions=self.num_actions)
        
    def degug_log(self):
        print("transition_count: ", self.transition_count)

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
        self.eval_net.optimizer.zero_grad()
        loss.backward()
        self.eval_net.optimizer.step()

        # update epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
