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

from whack_a_mole.utils import reseed
from whack_a_mole.create_env import create_env
from whack_a_mole.visualization import visualize, visualize_no_actor

from whack_a_mole.algorithms import TrainConfig, TrainResult, EvalResult, make_algorithm, QNet


def main():
    seed = 696
    env = create_env()
    reseed(seed, env)

    # TODO 2
    # Initialize and train the actors
    q_learning_config = TrainConfig(episodes=100,
                                    max_steps_per_episode=200,
                                    gamma=0.7, 
                                    learning_rate=0.01,
                                    seed=seed,
                                    device="cuda")
    q_net = QNet(action_dim=4, state_dim=6)
    q_actor = make_algorithm(name="q_learning", q_net=q_net)
    q_actor.train(env, q_learning_config)

    # TODO 3
    # Evaluate,
    # Visualize performance + videos
    visualize(env, q_actor, "q_test")

if __name__ == "__main__":
    main()


