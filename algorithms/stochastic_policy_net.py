import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as D
import torch

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