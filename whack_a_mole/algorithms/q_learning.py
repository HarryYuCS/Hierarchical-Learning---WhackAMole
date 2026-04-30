import torch
import torch.nn as nn
import numpy as np

class QNet(nn.Module):
    """
    A neural net to learn the Q function
    """
    def __init__(self, reward_dim : int, state_dim : int):
        """
        Initializes a neural net to approximate the q function given state and action
        
        Args:
            reward_dim : int
            state_dim : int
        """

    def forward(self, state, action):
        # TODO: define architecture and implement
        pass

class QLearningActor:
    """
    An actor using Q learning with sampling since we have a continuous action space
    """
    def __init__(self, q_net):
        self.q_net = q_net

    def compute_action(self, state):
        """
        Computes an action based on a given state, using sampling and our Q value approximation
        """
        # TODO
        pass

    def compute_loss(
        self,
        episode : list[tuple[np.ndarray, np.ndarray, float]],
        gamma : float
    ):
        # TODO
        pass
    
    def update_policy(self, episodes, optimizer, gamma):
        # TODO
        pass

    @property
    def device(self):
        return next(iter(self.policy_net.parameters())).device

    def to(self, device : str | torch.device):
        self.policy_net.to(device)
        return self



