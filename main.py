import mujoco
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv, MujocoPyFetchReachEnv
from torch.utils.data import DataLoader
import gymnasium.wrappers as wrappers
import matplotlib.pyplot as plt
import torch.distributions as D
from tqdm import tqdm, trange
import torch.optim as optim
import gymnasium_robotics
import gymnasium as gym
import torch.nn as nn
import numpy as np
import random
import torch

from utils import reseed
from create_env import create_env
from visualization import visualize, visualize_no_actor


def main():
    seed = 696
    env = create_env()
    reseed(seed, env)
    visualize(env)

    env = create_env(seed) # NOTE : don't know if this requires seed

    # TODO 2
    # Initialize and train the actors

    # TODO 3
    # Evaluate,
    # Visualize performance + videos

if __name__ == "__main__":
    main()


