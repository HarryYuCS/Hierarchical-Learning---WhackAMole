import torch
import random
import numpy as np
import gymnasium as gym

def reseed(seed, env=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if env is not None:
        env.unwrapped._np_random = gym.utils.seeding.np_random(seed)[0]