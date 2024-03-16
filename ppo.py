import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.layers(state)

# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.layers(state)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_param=0.2):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_param = clip_param

    def compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """
        Compute GAE-Lambda advantage and returns.
        """
        advantages = []
        last_adv = 0
        for t in reversed(range(len(rewards))): 
            if t + 1 < len(rewards):
                delta = rewards[t] + gamma * next_values[t] - values[t]
            else:
                delta = rewards[t] - values[t]
            last_adv = delta + gamma * lam * last_adv * (1 - dones[t])
            advantages.insert(0, last_adv)

        return advantages

    def update_networks(self, states, actions, log_probs_old, returns, advantages):
        """
        Perform PPO update on policy and value networks.
        """
        # Convert lists to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        log_probs_old = torch.tensor(log_probs_old)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.update_epochs):
            # Get current policy outputs
            log_probs = self.policy_net(states).log_prob(actions)
            state_values = self.value_net(states).squeeze()

            # Compute ratios
            ratios = torch.exp(log_probs - log_probs_old.detach())

            # Compute PPO objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = (returns - state_values).pow(2).mean()

            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()