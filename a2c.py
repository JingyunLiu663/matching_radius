import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the policy and value networks
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)


class A2CAgent:
    def __init__(self, input_dim, policy_hidden_dim, value_hidden_dim, action_dim, learning_rate=0.001, discount_factor=0.99):
        self.policy_net = PolicyNetwork(input_dim, policy_hidden_dim, action_dim)
        self.value_net = ValueNetwork(input_dim, value_hidden_dim)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor

    def choose_action(self, state: tuple):
        state_tensor = torch.tensor([state], dtype=torch.float32)  # Convert the state to a batch of size 1
        action_probs = self.policy_net(state_tensor).detach().numpy()[0]  # Get the first (and only) set of action probabilities
        return np.random.choice(len(action_probs), p=action_probs)

    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        # Convert to PyTorch tensors
        states = torch.tensor(batch_states, dtype=torch.float32)
        actions = torch.tensor(batch_actions, dtype=torch.long)
        rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        next_states = torch.tensor(batch_next_states, dtype=torch.float32)
        dones = torch.tensor(batch_dones, dtype=torch.float32)

        # Compute the estimated value and the advantages
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        target_values = rewards + self.discount_factor * next_values
        advantages = target_values - values

        # Policy loss
        action_probs = self.policy_net(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        policy_loss = -action_log_probs * advantages.detach()

        # Value loss
        value_loss = (values - target_values.detach()).pow(2)

        # Backpropagation
        self.policy_optimizer.zero_grad()
        policy_loss.mean().backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.mean().backward()
        self.value_optimizer.step()




# Example usage:
agent = A2CAgent(input_dim=10, hidden_dim=128, action_dim=3)

# Inside your training loop, call:
# agent.train(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
# action = agent.act(state)
