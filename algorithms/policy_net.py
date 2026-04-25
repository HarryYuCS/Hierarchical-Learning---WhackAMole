import torch.nn as nn
import torch

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Policy network for the REINFORCE algorithm.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(PolicyNet, self).__init__()
        # TODO 2.1a : define architecture
        

    def forward(self, state: torch.Tensor) -> D.Normal:
        """Forward pass of the policy network.

        Args:
            state (torch.Tensor): State of the environment. Shape (N, state_dim)

        Returns:
            action_dist (D.Normal): Normal distribution representing \pi(a_t | s_t)
        """
        # TODO 2.1b : implement forward  