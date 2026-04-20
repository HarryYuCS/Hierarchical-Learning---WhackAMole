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



seed = 695

reseed(seed)