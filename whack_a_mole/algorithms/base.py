from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainConfig:
    episodes: int = 200
    max_steps_per_episode: int = 200
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 8
    eval_every: int = 0
    eval_episodes: int = 5
    seed: int | None = None
    device: str = "cpu"


@dataclass
class TrainResult:
    episode_rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    timesteps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    mean_reward: float
    std_reward: float
    success_rate: float
    episode_rewards: list[float]


class RLAlgorithm(ABC):
    """
    Base abstract interface for an RL algorithm
    """
    @abstractmethod
    def train(self, env, config: TrainConfig) -> TrainResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, env, episodes: int = 10, deterministic: bool = True) -> EvalResult:
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, env=None):
        raise NotImplementedError
