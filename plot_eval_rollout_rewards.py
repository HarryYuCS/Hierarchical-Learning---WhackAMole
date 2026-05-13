from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from whack_a_mole.actors import SACActor, StitchedActor
from whack_a_mole.envs import create_env
from whack_a_mole.utils import reseed


def load_actor(args: argparse.Namespace, seed: int):
    if args.stitched:
        if args.task != "end_to_end":
            raise ValueError("--stitched requires --task end_to_end")
        if not args.pickup_checkpoint or not args.hammer_use_checkpoint:
            raise ValueError("--stitched requires --pickup-checkpoint and --hammer-use-checkpoint")

        pickup_env = create_env(render_mode=None, task="pickup")
        reseed(seed, pickup_env)
        pickup_actor = SACActor.load(args.pickup_checkpoint, env=pickup_env)
        pickup_env.close()

        hammer_env = create_env(render_mode=None, task="hammer_use")
        reseed(seed, hammer_env)
        hammer_actor = SACActor.load(args.hammer_use_checkpoint, env=hammer_env)
        hammer_env.close()

        return StitchedActor(pickup_actor=pickup_actor, hammer_use_actor=hammer_actor)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --stitched is used")

    env = create_env(render_mode=None, task=args.task)
    reseed(seed, env)
    actor = SACActor.load(args.checkpoint, env=env)
    env.close()
    return actor


def run_rollout(args: argparse.Namespace) -> tuple[list[int], list[float], float]:
    actor = load_actor(args, args.seed)
    env = create_env(render_mode=None, task=args.task)
    reseed(args.seed, env)

    obs, _ = env.reset()
    rewards: list[float] = []

    for step in range(1, args.max_steps + 1):
        action = actor.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break

    env.close()
    timesteps = list(range(1, len(rewards) + 1))
    total_reward = float(np.sum(rewards) if rewards else 0.0)
    return timesteps, rewards, total_reward


def save_plot(timesteps: list[int], rewards: list[float], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cumulative = np.cumsum(np.asarray(rewards, dtype=np.float64)) if rewards else np.array([], dtype=np.float64)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards, label="step reward", linewidth=1.8)
    if cumulative.size:
        plt.plot(timesteps, cumulative, label="cumulative reward", linewidth=1.5)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward vs timestep for one evaluation rollout")
    parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stitched", action="store_true")
    parser.add_argument("--pickup-checkpoint", default=None)
    parser.add_argument("--hammer-use-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=696)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--out", default="analysis/eval_rollout_reward_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    timesteps, rewards, total_reward = run_rollout(args)
    mode = "stitched" if args.stitched else "single"
    title = f"{mode} eval rollout reward vs timestep ({args.task})"
    out_png = Path(args.out)
    save_plot(timesteps, rewards, out_png, title)

    print(f"Saved plot: {out_png}")
    print(f"steps={len(rewards)} total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
