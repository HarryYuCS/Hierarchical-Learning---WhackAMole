from __future__ import annotations

import argparse
from pathlib import Path

from whack_a_mole.actors import SACActor, StitchedActor, TrainConfig
from whack_a_mole.envs import create_env
from whack_a_mole.evaluation import evaluate_actor
from whack_a_mole.training_viz import plot_training_metrics
from whack_a_mole.utils import reseed
from whack_a_mole.visualization import visualize, visualize_no_actor


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        seed=args.seed,
        show_progress=not args.no_progress,
        log_every=args.log_every,
    )


def default_ckpt_name(task: str, tag: str) -> str:
    return f"sac_dense_{task}_{tag}.zip"


def load_or_train_actor(
    *,
    task: str,
    seed: int,
    ckpt_dir: Path,
    ckpt_name: str,
    train_config: TrainConfig,
):
    env = create_env(render_mode=None, task=task)
    reseed(seed, env)
    actor = SACActor(environment=env)
    ckpt_path = ckpt_dir / ckpt_name

    if ckpt_path.exists():
        actor = SACActor.load(str(ckpt_path), env=env)
        print(f"Loaded checkpoint from {ckpt_path}")
        env.close()
        return actor

    result = actor.train(env, train_config)
    actor.save(str(ckpt_path))
    plot_path = ckpt_dir / f"{ckpt_path.stem}_metrics.png"
    plot_training_metrics(result, plot_path, title=f"SAC {task} training metrics")
    print(f"Saved training plot to {plot_path}")
    print(f"Trained {task}: timesteps={result.timesteps}")
    env.close()
    return actor


def load_actor_from_checkpoint(*, task: str, seed: int, checkpoint_path: str) -> SACActor:
    env = create_env(render_mode=None, task=task)
    reseed(seed, env)
    actor = SACActor.load(checkpoint_path, env=env)
    print(f"Loaded checkpoint for {task} from {checkpoint_path}")
    env.close()
    return actor


def inspect_stitched_switching(env, stitched_actor, episodes: int = 3):
    pickup_steps = 0
    hammer_use_steps = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(500):
            action = stitched_actor.predict(obs, deterministic=True)
            if stitched_actor.active_policy == "pickup":
                pickup_steps += 1
            else:
                hammer_use_steps += 1
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    print(
        f"Stitched policy usage: pickup_steps={pickup_steps}, "
        f"hammer_use_steps={hammer_use_steps}, switches={stitched_actor.switch_count}"
    )


def run_single_task(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = build_train_config(args)
    ckpt_name = args.checkpoint or default_ckpt_name(args.task, args.tag)

    actor = load_or_train_actor(
        task=args.task,
        seed=args.seed,
        ckpt_dir=ckpt_dir,
        ckpt_name=ckpt_name,
        train_config=config,
    )

    if args.visualize:
        video_env = create_env(render_mode="rgb_array", task=args.task)
        reseed(args.seed, video_env)
        video_name = args.video_name or f"sac_{args.task}_{args.tag}"
        visualize(video_env, actor, video_name, show_overlay=True)
        video_env.close()

    if args.evaluate_episodes > 0:
        eval_env = create_env(render_mode=None, task=args.task)
        reseed(args.seed, eval_env)
        eval_result = evaluate_actor(actor, eval_env, episodes=args.evaluate_episodes)
        eval_env.close()
        print(
            f"Task={args.task} success={eval_result.success_rate:.3f}, "
            f"mean_ep_reward={eval_result.mean_episode_reward:.2f}, "
            f"mean_step_reward={eval_result.mean_step_reward:.3f}, "
            f"mean_tip_dist={eval_result.mean_tip_distance:.3f}"
        )


def run_full_pipeline(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in [
            ("--pickup-checkpoint", args.pickup_checkpoint),
            ("--hammer-use-checkpoint", args.hammer_use_checkpoint),
            ("--e2e-checkpoint", args.e2e_checkpoint),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Full pipeline requires explicit checkpoints for all parts. Missing: "
            + ", ".join(missing)
        )

    pickup_actor = load_actor_from_checkpoint(
        task="pickup", seed=args.seed, checkpoint_path=args.pickup_checkpoint
    )
    hammer_use_actor = load_actor_from_checkpoint(
        task="hammer_use", seed=args.seed, checkpoint_path=args.hammer_use_checkpoint
    )
    e2e_actor = load_actor_from_checkpoint(
        task="end_to_end", seed=args.seed, checkpoint_path=args.e2e_checkpoint
    )

    stitched_actor = StitchedActor(pickup_actor=pickup_actor, hammer_use_actor=hammer_use_actor)
    eval_env = create_env(render_mode=None, task="end_to_end")
    reseed(args.seed, eval_env)
    inspect_stitched_switching(eval_env, stitched_actor, episodes=3)
    stitched_eval = evaluate_actor(stitched_actor, eval_env, episodes=args.evaluate_episodes)
    e2e_eval = evaluate_actor(e2e_actor, eval_env, episodes=args.evaluate_episodes)
    eval_env.close()

    print(
        f"Stitched success={stitched_eval.success_rate:.3f}, "
        f"mean_ep_reward={stitched_eval.mean_episode_reward:.2f}, "
        f"mean_step_reward={stitched_eval.mean_step_reward:.3f}, "
        f"mean_tip_dist={stitched_eval.mean_tip_distance:.3f}; "
        f"End-to-end success={e2e_eval.success_rate:.3f}, "
        f"mean_ep_reward={e2e_eval.mean_episode_reward:.2f}, "
        f"mean_step_reward={e2e_eval.mean_step_reward:.3f}, "
        f"mean_tip_dist={e2e_eval.mean_tip_distance:.3f}"
    )

    if args.visualize:
        visuals = [
            ("pickup", pickup_actor, "sac_pickup_only"),
            ("hammer_use", hammer_use_actor, "sac_hammer_use_only"),
            ("end_to_end", stitched_actor, "sac_stitched_end_to_end"),
            ("end_to_end", e2e_actor, "sac_e2e_end_to_end"),
        ]
        for task, actor, video_name in visuals:
            video_env = create_env(render_mode="rgb_array", task=task)
            reseed(args.seed, video_env)
            visualize(video_env, actor, video_name, show_overlay=True)
            video_env.close()


def run_see_envs(args: argparse.Namespace) -> None:
    env = create_env(render_mode="rgb_array", task=args.task)
    visualize_no_actor(env, video_name=f"{args.task}_env")
    env.close()


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Whack-a-mole training/evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("help", help="Show command help and usage guide")

    train_parser = subparsers.add_parser("train", help="Train one task or full pipeline")
    train_parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], default="hammer_use")
    train_parser.add_argument("--pipeline", choices=["single", "full"], default="single")
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument("--pickup-checkpoint", default=None)
    train_parser.add_argument("--hammer-use-checkpoint", default=None)
    train_parser.add_argument("--e2e-checkpoint", default=None)
    train_parser.add_argument("--video-name", default=None)
    train_parser.add_argument("--seed", type=int, default=696)
    train_parser.add_argument("--checkpoint-dir", default="model_checkpoints")
    train_parser.add_argument("--tag", default="v1")
    train_parser.add_argument("--episodes", type=int, default=400)
    train_parser.add_argument("--max-steps", type=int, default=50)
    train_parser.add_argument("--gamma", type=float, default=0.95)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--log-every", type=int, default=5)
    train_parser.add_argument("--evaluate-episodes", type=int, default=10)
    train_parser.add_argument("--visualize", action="store_true")
    train_parser.add_argument("--no-progress", action="store_true")

    envs_parser = subparsers.add_parser("see-envs", help="Render a raw environment video")
    envs_parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], default="hammer_use")
    envs_parser.add_argument("--seed", type=int, default=696)

    parser.set_defaults(command="help")
    return parser, parser.parse_args()


def print_usage_guide() -> None:
    print("Usage guide:")
    print("  1) Train/eval one task (defaults: SAC + hammer_use):")
    print("     python -m whack_a_mole train --visualize")
    print("  2) Train/eval non-stitched end-to-end separately:")
    print("     python -m whack_a_mole train --task end_to_end --visualize")
    print("  3) Run full stitched-vs-e2e evaluation from checkpoints:")
    print(
        "     python -m whack_a_mole train --pipeline full "
        "--pickup-checkpoint model_checkpoints/sac_dense_pickup_v1.zip "
        "--hammer-use-checkpoint model_checkpoints/sac_dense_hammer_use_v1.zip "
        "--e2e-checkpoint model_checkpoints/sac_dense_end_to_end_v1.zip --visualize"
    )
    print("  4) Render an environment without an actor:")
    print("     python -m whack_a_mole see-envs --task hammer_use")
    print("  5) Override checkpoints/config:")
    print("     --checkpoint-dir model_checkpoints --tag v2 --checkpoint custom_name.zip")


def main() -> None:
    parser, args = parse_args()
    if args.command == "train":
        if args.pipeline == "single":
            run_single_task(args)
        elif args.pipeline == "full":
            run_full_pipeline(args)
        else:
            raise ValueError(f"Unsupported pipeline {args.pipeline}")
    elif args.command == "see-envs":
        run_see_envs(args)
    elif args.command == "help":
        parser.print_help()
        print()
        print_usage_guide()
    else:
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
