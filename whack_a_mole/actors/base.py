from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainConfig:
    """Configuration used when training an actor.

    Attributes:
        episodes: Number of episodes used for training.
        max_steps_per_episode: Maximum number of steps per episode.
        gamma: Discount factor.
        learning_rate: Optimizer learning rate.
        batch_size: Batch size for updates when applicable.
        eval_every: Episode frequency for periodic evaluation.
        eval_episodes: Number of episodes per periodic evaluation.
        seed: Optional random seed.
        device: Compute device hint.
        show_progress: Whether to print progress from the trainer.
        log_every: Frequency for logging summaries.
    """
    episodes: int = 200
    max_steps_per_episode: int = 200
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 8
    eval_every: int = 0
    eval_episodes: int = 5
    seed: int | None = None
    device: str = "cpu"
    show_progress: bool = True
    log_every: int = 10


@dataclass
class TrainResult:
    """Summary metrics returned from training.

    Attributes:
        episode_rewards: Per-episode reward history.
        losses: Per-update loss history.
        timesteps: Total environment timesteps consumed.
        metadata: Additional algorithm-specific fields.
    """
    episode_rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    timesteps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Summary metrics returned from evaluation.

    Attributes:
        mean_reward: Mean episode reward.
        std_reward: Standard deviation of episode reward.
        success_rate: Fraction of successful episodes.
        episode_rewards: Raw episode reward values.
    """
    mean_reward: float
    std_reward: float
    success_rate: float
    episode_rewards: list[float]


class Actor(ABC):
    """Inference-only actor interface."""

    @abstractmethod
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        """Predict an action from an observation.

        Args:
            obs: Environment observation.
            deterministic: Whether to use deterministic action selection.

        Returns:
            Predicted action compatible with the environment action space.
        """
        raise NotImplementedError


class TrainableActor(Actor, ABC):
    """Actor interface that supports training and checkpointing."""

    @abstractmethod
    def train(self, env, config: TrainConfig) -> TrainResult:
        """Train the actor on an environment.

        Args:
            env: Training environment.
            config: Training hyperparameters.

        Returns:
            Aggregated training metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, env, episodes: int = 10, deterministic: bool = True) -> EvalResult:
        """Evaluate the actor policy.

        Args:
            env: Evaluation environment.
            episodes: Number of episodes to evaluate.
            deterministic: Whether to use deterministic actions.

        Returns:
            Aggregated evaluation metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save actor parameters to disk.

        Args:
            path: Output checkpoint path.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, env=None):
        """Load actor parameters from disk.

        Args:
            path: Input checkpoint path.
            env: Optional environment binding.

        Returns:
            Instantiated actor loaded from checkpoint.
        """
        raise NotImplementedError
