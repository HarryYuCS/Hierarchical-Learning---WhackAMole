import torch.nn as nn


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

    def forward(state, action):
        # TODO: define architecture and implement
        pass

class QLearningActor:
    """
    An actor using Q learning with sampling since we have a continuous action space
    """
    def __init__(self, q_net):
        self.q_net = q_net
    
    def update_policy(self, episodes, optimizer, gamma):



