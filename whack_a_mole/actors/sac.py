from __future__ import annotations

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from whack_a_mole.actors.base import EvalResult, TrainConfig, TrainResult, TrainableActor
from whack_a_mole.actors.callbacks import SB3MetricsCallback


class SACActor(TrainableActor):
    """SAC-backed trainable actor using Stable-Baselines3."""

    def __init__(self, ckpt: str = None, environment=None, model=None, policy: str = "MultiInputPolicy"):
        self.environment = environment
        self.policy = policy

        if model is not None:
            self.model = model
            return

        if ckpt is not None:
            self.model = SAC.load(ckpt, environment)
        else:
            self.model = None

    def predict(self, obs, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def train(self, env, config: TrainConfig) -> TrainResult:
        if not isinstance(env, gym.wrappers.TimeLimit):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps_per_episode)
        if not isinstance(env, Monitor):
            env = Monitor(env)
        self.environment = env
        total_timesteps = config.episodes * config.max_steps_per_episode
        if self.model is None:
            self.model = SAC(
                self.policy,
                env,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                seed=config.seed,
                buffer_size=300_000,
                learning_starts=1_000,
                batch_size=256,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                verbose=1 if config.show_progress else 0,
            )
        else:
            self.model.set_env(env)
        callback = SB3MetricsCallback(
            metric_keys=["train/actor_loss", "train/critic_loss", "train/ent_coef", "rollout/ep_rew_mean"]
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        critic_series = callback.metrics.get("train/critic_loss", {"values": []}).get("values", [])
        losses = [x for x in critic_series if not np.isnan(x)]
        ep_rew_series = callback.metrics.get("rollout/ep_rew_mean", {"values": []}).get("values", [])
        return TrainResult(
            episode_rewards=[x for x in ep_rew_series if not np.isnan(x)],
            losses=losses,
            timesteps=total_timesteps,
            metadata={
                "algorithm": "sac",
                "metrics": callback.metrics,
            },
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
        model = SAC.load(path, env)
        return cls(model=model, environment=env)
