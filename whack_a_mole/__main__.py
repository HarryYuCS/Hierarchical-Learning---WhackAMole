from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from whack_a_mole.actors import StitchedActor
from whack_a_mole.envs import create_env
from whack_a_mole.evaluation import evaluate_actor
from whack_a_mole.utils import (
    build_train_config,
    default_ckpt_name,
    load_actor_from_checkpoint,
    load_or_train_actor,
    reseed,
)
from whack_a_mole.visualization import visualize, visualize_no_actor


def run_train(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = build_train_config(args)
    ckpt_name = args.checkpoint or default_ckpt_name(args.task)

    actor = load_or_train_actor(
        task=args.task,
        seed=args.seed,
        ckpt_dir=ckpt_dir,
        ckpt_name=ckpt_name,
        train_config=config,
        plot_train_metrics=args.plot_train_metrics,
    )

    if args.visualize:
        video_env = create_env(render_mode="rgb_array", task=args.task)
        reseed(args.seed, video_env)
        video_name = args.video_name or f"sac_{args.task}"
        visualize(video_env, actor, video_name, show_overlay=True)
        video_env.close()


def run_see_envs(args: argparse.Namespace) -> None:
    env = create_env(render_mode="rgb_array", task=args.task)
    visualize_no_actor(env, video_name=f"{args.task}_env")
    env.close()


def run_evaluate(args: argparse.Namespace) -> None:
    seeds = [args.seed + i for i in range(args.num_seeds)]
    results = []

    def build_actor_for_seed(seed: int):
        if args.stitched:
            if args.task != "end_to_end":
                raise ValueError("--stitched evaluation requires --task end_to_end")
            if not args.pickup_checkpoint or not args.hammer_use_checkpoint:
                raise ValueError("--stitched requires --pickup-checkpoint and --hammer-use-checkpoint")
            pickup_actor = load_actor_from_checkpoint(
                task="pickup", seed=seed, checkpoint_path=args.pickup_checkpoint
            )
            hammer_use_actor = load_actor_from_checkpoint(
                task="hammer_use", seed=seed, checkpoint_path=args.hammer_use_checkpoint
            )
            return StitchedActor(pickup_actor=pickup_actor, hammer_use_actor=hammer_use_actor)

        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            raise ValueError("evaluate requires --checkpoint unless --stitched is used")
        return load_actor_from_checkpoint(task=args.task, seed=seed, checkpoint_path=checkpoint_path)

    def rollout_rewards(actor, seed: int) -> list[float]:
        env = create_env(render_mode=None, task=args.task)
        reseed(seed, env)
        obs, _ = env.reset()
        rewards: list[float] = []
        for _ in range(args.eval_max_steps):
            action = actor.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        env.close()
        return rewards

    def save_rollout_plot(rewards: list[float], seed: int) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(args.plot_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timesteps = np.arange(1, len(rewards) + 1, dtype=np.int32)
        cumulative = np.cumsum(np.asarray(rewards, dtype=np.float64))
        mode = "stitched" if args.stitched else "single"
        out_path = out_dir / f"{mode}_{args.task}_seed{seed}_reward_curve.png"

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, rewards, label="step reward", linewidth=1.8)
        plt.plot(timesteps, cumulative, label="cumulative reward", linewidth=1.5)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title(f"{mode} evaluate rollout ({args.task}, seed={seed})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
        print(f"Saved rollout reward plot: {out_path}")

    for seed in seeds:
        actor = build_actor_for_seed(seed)

        eval_env = create_env(render_mode=None, task=args.task)
        reseed(seed, eval_env)
        eval_result = evaluate_actor(
            actor,
            eval_env,
            episodes=args.evaluate_episodes,
            max_steps=args.eval_max_steps,
        )
        eval_env.close()
        results.append((seed, eval_result))

        if args.visualize:
            video_env = create_env(render_mode="rgb_array", task=args.task)
            reseed(seed, video_env)
            mode = "stitched" if args.stitched else "single"
            video_name = f"{args.video_name_prefix}_{mode}_{args.task}_seed{seed}"
            visualize(video_env, actor, video_name, show_overlay=True, max_steps=args.video_max_steps)
            video_env.close()

        if args.plot_rollout_reward:
            save_rollout_plot(rollout_rewards(actor, seed), seed)

    ckpt_desc = (
        f"pickup={args.pickup_checkpoint}, hammer_use={args.hammer_use_checkpoint}"
        if args.stitched
        else f"checkpoint={args.checkpoint}"
    )
    print(
        f"Evaluate task={args.task} {ckpt_desc} episodes_per_seed={args.evaluate_episodes} "
        f"max_steps={args.eval_max_steps} seeds={seeds}"
    )
    for seed, eval_result in results:
        print(
            f"seed={seed} success={eval_result.success_rate:.3f} "
            f"mean_ep_reward={eval_result.mean_episode_reward:.2f} "
            f"mean_step_reward={eval_result.mean_step_reward:.3f} "
            f"mean_tip_dist={eval_result.mean_tip_distance:.3f}"
        )

    success_values = [r.success_rate for _, r in results]
    ep_reward_values = [r.mean_episode_reward for _, r in results]
    step_reward_values = [r.mean_step_reward for _, r in results]
    tip_dist_values = [r.mean_tip_distance for _, r in results]

    def mean_std(values):
        return float(sum(values) / len(values)), float((sum((x - (sum(values) / len(values))) ** 2 for x in values) / len(values)) ** 0.5)

    success_mean, success_std = mean_std(success_values)
    ep_reward_mean, ep_reward_std = mean_std(ep_reward_values)
    step_reward_mean, step_reward_std = mean_std(step_reward_values)
    tip_dist_mean, tip_dist_std = mean_std(tip_dist_values)

    print(
        "aggregate "
        f"success_mean={success_mean:.3f} success_std={success_std:.3f} "
        f"mean_ep_reward={ep_reward_mean:.2f} +/- {ep_reward_std:.2f} "
        f"mean_step_reward={step_reward_mean:.3f} +/- {step_reward_std:.3f} "
        f"mean_tip_dist={tip_dist_mean:.3f} +/- {tip_dist_std:.3f}"
    )


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Whack-a-mole training/evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("help", help="Show command help and usage guide")

    train_parser = subparsers.add_parser("train", help="Train/load one task actor")
    train_parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], default="hammer_use")
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument("--video-name", default=None)
    train_parser.add_argument("--seed", type=int, default=696)
    train_parser.add_argument("--checkpoint-dir", default="model_checkpoints")
    train_parser.add_argument("--episodes", type=int, default=400)
    train_parser.add_argument("--max-steps", type=int, default=50)
    train_parser.add_argument("--gamma", type=float, default=0.95)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--log-every", type=int, default=5)
    train_parser.add_argument("--visualize", action="store_true")
    train_parser.add_argument("--plot-train-metrics", action="store_true")
    train_parser.add_argument("--no-progress", action="store_true")

    envs_parser = subparsers.add_parser("see-envs", help="Render a raw environment video")
    envs_parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], default="hammer_use")
    envs_parser.add_argument("--seed", type=int, default=696)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint across multiple seeds")
    eval_parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], required=True)
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--stitched", action="store_true")
    eval_parser.add_argument("--pickup-checkpoint", default=None)
    eval_parser.add_argument("--hammer-use-checkpoint", default=None)
    eval_parser.add_argument("--seed", type=int, default=696)
    eval_parser.add_argument("--num-seeds", type=int, default=1)
    eval_parser.add_argument("--evaluate-episodes", type=int, default=10)
    eval_parser.add_argument("--eval-max-steps", type=int, default=200)
    eval_parser.add_argument("--visualize", action="store_true")
    eval_parser.add_argument("--video-name-prefix", default="evaluate")
    eval_parser.add_argument("--video-max-steps", type=int, default=200)
    eval_parser.add_argument("--plot-rollout-reward", action="store_true")
    eval_parser.add_argument("--plot-out-dir", default="analysis/evaluate")

    parser.set_defaults(command="help")
    return parser, parser.parse_args()


def print_usage_guide() -> None:
    print("Usage guide:")
    print("  1) Train one task (defaults: SAC + hammer_use):")
    print("     python -m whack_a_mole train --visualize")
    print("  2) Train non-stitched end-to-end actor:")
    print("     python -m whack_a_mole train --task end_to_end --plot-train-metrics")
    print("  3) Render an environment without an actor:")
    print("     python -m whack_a_mole see-envs --task hammer_use")
    print("  4) Evaluate a checkpoint across seeds:")
    print("     python -m whack_a_mole evaluate --task hammer_use --checkpoint model_checkpoints/sac_dense_hammer_use_v1.zip --num-seeds 5")
    print("  5) Evaluate stitched actor across seeds:")
    print("     python -m whack_a_mole evaluate --stitched --task end_to_end --pickup-checkpoint model_checkpoints/sac_dense_pickup_v1.zip --hammer-use-checkpoint model_checkpoints/sac_dense_hammer_use_v1.zip --num-seeds 5")
    print("  6) Evaluate with visualization and reward-curve plots:")
    print("     python -m whack_a_mole evaluate --task hammer_use --checkpoint model_checkpoints/sac_dense_hammer_use_v1.zip --visualize --plot-rollout-reward")
    print("  7) Override checkpoint path:")
    print("     --checkpoint-dir model_checkpoints --checkpoint custom_name.zip")


def main() -> None:
    parser, args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "see-envs":
        run_see_envs(args)
    elif args.command == "evaluate":
        if args.num_seeds <= 0:
            raise ValueError("--num-seeds must be positive")
        if args.evaluate_episodes <= 0:
            raise ValueError("--evaluate-episodes must be positive")
        if args.eval_max_steps <= 0:
            raise ValueError("--eval-max-steps must be positive")
        if args.video_max_steps <= 0:
            raise ValueError("--video-max-steps must be positive")
        run_evaluate(args)
    elif args.command == "help":
        parser.print_help()
        print()
        print_usage_guide()
    else:
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
