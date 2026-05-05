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
from pathlib import Path

from whack_a_mole.utils import reseed
from whack_a_mole.create_env import create_env
from whack_a_mole.visualization import visualize, visualize_no_actor

from whack_a_mole.algorithms import TrainConfig, make_algorithm, QNet, QLearningActor


def main():
    seed = 696
    train_env = create_env(render_mode=None, reward_type="dense")
    reseed(seed, train_env)

    # TODO 2
    # Initialize and train the actors
    q_learning_config = TrainConfig(episodes=100,
                                    max_steps_per_episode=200,
                                    gamma=0.7, 
                                    learning_rate=0.01,
                                    seed=seed,
                                    device="cuda",
                                    show_progress=True,
                                    log_every=5)
    initial_obs, _ = train_env.reset(seed=seed)
    state_dim = int(np.asarray(initial_obs["observation"]).shape[0])
    action_dim = int(np.prod(train_env.action_space.shape))
    q_net = QNet(action_dim=action_dim, state_dim=state_dim)
    q_actor = make_algorithm(name="q_learning", q_net=q_net)

    ckpt_path = Path("model_checkpoints/q_learning_dense.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists():
        q_actor = QLearningActor.load(str(ckpt_path), env=train_env)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        train_result = q_actor.train(train_env, q_learning_config)
        q_actor.save(str(ckpt_path))
        print(
            f"Training complete: episodes={len(train_result.episode_rewards)} timesteps={train_result.timesteps}"
        )

    # TODO 3
    # Evaluate,
    # Visualize performance + videos
    train_env.close()
    video_env = create_env(render_mode="rgb_array")
    reseed(seed, video_env)
    visualize(video_env, q_actor, "q_test_dense")

if __name__ == "__main__":
    main()
