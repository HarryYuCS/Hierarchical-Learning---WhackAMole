
import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as D
import torch
import numpy as np

# TODO 2.1c : implement REINFORCE

class StochasticPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Policy network for the REINFORCE algorithm.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(StochasticPolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim * 2)

        

    def forward(self, state: torch.Tensor) -> D.Normal:
        """Forward pass of the policy network.

        Args:
            state (torch.Tensor): State of the environment. Shape (N, state_dim)

        Returns:
            action_dist (D.Normal): Normal distribution representing \pi(a_t | s_t)
        """
        x = f.relu(self.fc1(state.float()))
        x = self.fc2(x)
        mu, ln_sigma = torch.split(x, x.shape[-1]//2, -1)

        return D.Normal(mu, torch.exp(ln_sigma))


class Reinforce:
    def __init__(self, policy_net : StochasticPolicyNet):
        """Policy gradient algorithm based on the REINFORCE algorithm, with REWARD-TO-GO

        Args:
            policy_net (PolicyNet): Policy networks
            reward_to_go (bool): True if using reward_to_go, False if not (False in part 1, True in part 2)
        """
        self.policy_net = policy_net

    def compute_action(self, state : np.ndarray) -> np.ndarray:
        """Select an action based on the policy network

        Args:
            state (np.ndarray): State of the environment

        Returns:
            action (np.ndarray): Action to take
        """
        with torch.no_grad():
          normal_tensor = self.policy_net(torch.from_numpy(state))
          action = normal_tensor.sample()

          return action.numpy()

    def compute_loss(
      self,
      episode : list[tuple[np.ndarray, np.ndarray, float]],
      gamma : float
    ) -> torch.Tensor:
        """Compute the loss function J for the REINFORCE algorithm

        Args:
            episode (list): List of tuples (state, action, reward)
            gamma (float): Discount factor

        Returns:
            loss (torch.Tensor): The value of the loss function
        """
        states, actions, rewards = zip(*episode)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).float()

        normal = self.policy_net(states)
        log_probs = normal.log_prob(actions).sum(dim=-1)

        # Reward to go with DP
        T = len(episode)
        dp = np.zeros(T + 1)
        for timestep in range(T - 1, -1, -1):
          current_reward = rewards[timestep] + gamma * dp[timestep + 1]
          dp[timestep] = current_reward
        rewards = torch.from_numpy(dp[:-1]).float()

        loss = -(log_probs * rewards).sum()

        return loss

    def update_policy(self, episodes, optimizer, gamma):
        """Update the policy network using the batch of episodes

        Args:
            episodes (list): List of episodes
            optimizer (torch.optim): Optimizer
            gamma (float): Discount factor
        Returns:
            loss (float): The value of the loss function
        """
        losses = []
        for episode in episodes:
          losses.append(self.compute_loss(episode, gamma))

        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    @property
    def device(self):
        return next(iter(self.policy_net.parameters())).device

    def to(self, device : str | torch.device):
        self.policy_net.to(device)
        return self