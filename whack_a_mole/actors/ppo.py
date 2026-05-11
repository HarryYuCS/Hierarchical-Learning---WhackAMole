from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from whack_a_mole.actors.base import EvalResult, TrainConfig, TrainResult, TrainableActor
from whack_a_mole.actors.callbacks import SB3MetricsCallback


class PPOActor(TrainableActor):
    """PPO-backed trainable actor using Stable-Baselines3.

    This class wraps a SB3 `PPO` model and exposes the project-level
    train/evaluate/predict/save/load interface.

    Attributes:
        environment: Bound environment used by the underlying model.
        policy: SB3 policy class name.
        model: Underlying `stable_baselines3.PPO` instance.
    """

    def __init__(self, ckpt: str = None, environment=None, model=None, policy: str = "MultiInputPolicy"):
        """Initialize a PPO actor.

        Args:
            ckpt: Optional checkpoint path to load.
            environment: Optional environment to bind.
            model: Optional pre-created SB3 model.
            policy: SB3 policy name used when creating a fresh model.
        """
        self.environment = environment
        self.policy = policy

        if model is not None:
            self.model = model
            return

        if ckpt is not None:
            self.model = PPO.load(ckpt, environment)
        else:
            self.model = None

    def predict(self, obs, deterministic: bool = True):
        """Predict an action from observation.

        Args:
            obs: Environment observation.
            deterministic: Whether to use deterministic action selection.

        Returns:
            Action predicted by PPO.
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def train(self, env, config: TrainConfig) -> TrainResult:
        """Train PPO for a fixed number of timesteps.

        Args:
            env: Training environment.
            config: Training hyperparameters.

        Returns:
            Training summary with total timesteps.
        """
        if not isinstance(env, Monitor):
            env = Monitor(env)
        self.environment = env
        total_timesteps = config.episodes * config.max_steps_per_episode
        if self.model is None:
            self.model = PPO(
                self.policy,
                env,
                gamma=config.gamma,
                learning_rate=config.learning_rate,
                seed=config.seed,
                verbose=1 if config.show_progress else 0,
                n_steps=1024,
                batch_size=64,
            )
        else:
            self.model.set_env(env)
        callback = SB3MetricsCallback(
            metric_keys=["train/loss", "train/value_loss", "train/policy_gradient_loss", "rollout/ep_rew_mean"]
        )
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        loss_series = callback.metrics.get("train/loss", {"values": []}).get("values", [])
        losses = [x for x in loss_series if not np.isnan(x)]
        ep_rew_series = callback.metrics.get("rollout/ep_rew_mean", {"values": []}).get("values", [])
        return TrainResult(
            episode_rewards=[x for x in ep_rew_series if not np.isnan(x)],
            losses=losses,
            timesteps=total_timesteps,
            metadata={
                "algorithm": "ppo",
                "metrics": callback.metrics,
            },
        )

    def evaluate(self, env, episodes: int = 10, deterministic: bool = True) -> EvalResult:
        """Evaluate PPO over multiple episodes.

        Args:
            env: Evaluation environment.
            episodes: Number of episodes to evaluate.
            deterministic: Whether to use deterministic actions.

        Returns:
            Aggregated evaluation statistics.
        """
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
        """Save PPO checkpoint.

        Args:
            path: Output checkpoint path.
        """
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None):
        """Load PPO checkpoint.

        Args:
            path: Input checkpoint path.
            env: Optional environment to bind.

        Returns:
            Loaded PPO actor instance.
        """
        model = PPO.load(path, env)
        return cls(model=model, environment=env)
