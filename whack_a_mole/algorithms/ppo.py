from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from whack_a_mole.algorithms.base import EvalResult, RLAlgorithm, TrainConfig, TrainResult


class PPOActor(RLAlgorithm):
    def __init__(self, ckpt: str=None, environment=None, model=None):
        '''
          Requires environment to be a 1-vectorized environment

          The `ckpt` is a .zip file path that leads to the checkpoint you want
          to use for this particular actor.

          If the `model` variable is provided, then this constructor will store
          that as the internal representing model instead of loading one from the
          checkpoint path
        '''
        assert ckpt is not None or model is not None

        if model is not None:
            self.model = model
            self.environment = environment
            return

        self.model = PPO.load(ckpt, environment)
        self.environment = environment

    def predict(self, obs, deterministic: bool = True):
        '''Gives the action prediction of this particular actor'''
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def train(self, env, config: TrainConfig) -> TrainResult:
        self.environment = env
        total_timesteps = config.episodes * config.max_steps_per_episode
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps)
        return TrainResult(
            episode_rewards=[],
            losses=[],
            timesteps=total_timesteps,
            metadata={"algorithm": "ppo"},
        )

    def evaluate(self, env, episodes: int = 10, deterministic: bool = True) -> EvalResult:
        rewards = []
        successes = []
        for _ in range(episodes):
            obs, _ = env.reset()
            total = 0.0
            final_success = 0.0
            for _ in range(500):
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                final_success = float(info.get("is_success", 0.0))
                if terminated or truncated:
                    break
            rewards.append(total)
            successes.append(final_success)

        return EvalResult(
            mean_reward=float(np.mean(rewards) if rewards else 0.0),
            std_reward=float(np.std(rewards) if rewards else 0.0),
            success_rate=float(np.mean(successes) if successes else 0.0),
            episode_rewards=rewards,
        )

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None):
        model = PPO.load(path, env)
        return cls(model=model, environment=env)
