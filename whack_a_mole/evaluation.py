from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalSummary:
    episodes: int
    mean_episode_reward: float
    std_episode_reward: float
    mean_step_reward: float
    success_rate: float
    mean_tip_distance: float


def evaluate_actor(actor, env, episodes: int = 20, max_steps: int = 500) -> EvalSummary:
    """Evaluate an actor on an environment over multiple episodes.

    Args:
        actor: Policy object exposing ``predict(obs, deterministic=True)``.
        env: Evaluation environment.
        episodes: Number of episodes to sample.
        max_steps: Maximum steps per episode.

    Returns:
        Aggregate evaluation summary across sampled episodes.
    """
    episode_rewards: list[float] = []
    step_rewards: list[float] = []
    successes: list[float] = []
    tip_distances: list[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        final_success = 0.0
        for _ in range(max_steps):
            action = actor.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            r = float(reward)
            total += r
            step_rewards.append(r)
            final_success = float(info.get("is_success", 0.0))
            if "hammer_tip_distance" in info:
                tip_distances.append(float(info["hammer_tip_distance"]))
            if terminated or truncated:
                break

        episode_rewards.append(total)
        successes.append(final_success)

    return EvalSummary(
        episodes=episodes,
        mean_episode_reward=float(np.mean(episode_rewards) if episode_rewards else 0.0),
        std_episode_reward=float(np.std(episode_rewards) if episode_rewards else 0.0),
        mean_step_reward=float(np.mean(step_rewards) if step_rewards else 0.0),
        success_rate=float(np.mean(successes) if successes else 0.0),
        mean_tip_distance=float(np.mean(tip_distances) if tip_distances else float("nan")),
    )
