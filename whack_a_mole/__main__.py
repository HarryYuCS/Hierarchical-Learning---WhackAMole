from __future__ import annotations

import argparse
from pathlib import Path

from whack_a_mole.actors import PPOActor, SACActor, StitchedPPOActor, TrainConfig
from whack_a_mole.create_env import create_env
from whack_a_mole.evaluation import evaluate_actor
from whack_a_mole.training_viz import plot_training_metrics
from whack_a_mole.utils import reseed
from whack_a_mole.visualization import visualize, visualize_no_actor


def model_matches_env(actor, env) -> bool:
    model = getattr(actor, "model", None)
    if model is None:
        return False
    try:
        return model.observation_space == env.observation_space and model.action_space == env.action_space
    except Exception:
        return False


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


def default_ckpt_name(algo: str, task: str, tag: str) -> str:
    return f"{algo}_dense_{task}_{tag}.zip"


def load_or_train_actor(
    *,
    algo: str,
    task: str,
    seed: int,
    ckpt_dir: Path,
    ckpt_name: str,
    train_config: TrainConfig,
):
    env = create_env(render_mode=None, task=task)
    reseed(seed, env)
    actor_cls = PPOActor if algo == "ppo" else SACActor
    actor = actor_cls(environment=env)
    ckpt_path = ckpt_dir / ckpt_name

    if ckpt_path.exists():
        actor = actor_cls.load(str(ckpt_path), env=env)
        if algo == "ppo" and not model_matches_env(actor, env):
            print(f"Checkpoint {ckpt_path} mismatched env; retraining")
        else:
            print(f"Loaded checkpoint from {ckpt_path}")
            env.close()
            return actor

    result = actor.train(env, train_config)
    actor.save(str(ckpt_path))
    plot_path = ckpt_dir / f"{ckpt_path.stem}_metrics.png"
    plot_training_metrics(result, plot_path, title=f"{algo.upper()} {task} training metrics")
    print(f"Saved training plot to {plot_path}")
    print(f"Trained {task}: timesteps={result.timesteps}")
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
    ckpt_name = args.checkpoint or default_ckpt_name(args.algo, args.task, args.tag)

    actor = load_or_train_actor(
        algo=args.algo,
        task=args.task,
        seed=args.seed,
        ckpt_dir=ckpt_dir,
        ckpt_name=ckpt_name,
        train_config=config,
    )

    if args.visualize:
        video_env = create_env(render_mode="rgb_array", task=args.task)
        reseed(args.seed, video_env)
        video_name = args.video_name or f"{args.algo}_{args.task}_{args.tag}"
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
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = build_train_config(args)

    pickup_actor = load_or_train_actor(
        algo="ppo",
        task="pickup",
        seed=args.seed,
        ckpt_dir=ckpt_dir,
        ckpt_name=default_ckpt_name("ppo", "pickup", args.tag),
        train_config=config,
    )
    hammer_use_actor = load_or_train_actor(
        algo="ppo",
        task="hammer_use",
        seed=args.seed,
        ckpt_dir=ckpt_dir,
        ckpt_name=default_ckpt_name("ppo", "hammer_use", args.tag),
        train_config=config,
    )
    e2e_actor = load_or_train_actor(
        algo="ppo",
        task="end_to_end",
        seed=args.seed,
        ckpt_dir=ckpt_dir,
        ckpt_name=default_ckpt_name("ppo", "end_to_end", args.tag),
        train_config=config,
    )

    stitched_actor = StitchedPPOActor(pickup_actor=pickup_actor, hammer_use_actor=hammer_use_actor)
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
            ("pickup", pickup_actor, "ppo_pickup_only"),
            ("hammer_use", hammer_use_actor, "ppo_hammer_use_only"),
            ("end_to_end", stitched_actor, "ppo_stitched_end_to_end"),
            ("end_to_end", e2e_actor, "ppo_e2e_end_to_end"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whack-a-mole training/evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=False)

    train_parser = subparsers.add_parser("train", help="Train/load one task with PPO or SAC")
    train_parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    train_parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], default="hammer_use")
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument("--video-name", default=None)

    full_parser = subparsers.add_parser("full", help="Run full PPO stitched vs end-to-end pipeline")

    envs_parser = subparsers.add_parser("see-envs", help="Render a raw environment video")
    envs_parser.add_argument("--task", choices=["pickup", "hammer_use", "end_to_end"], default="hammer_use")

    for sub in [train_parser, full_parser, envs_parser]:
        sub.add_argument("--seed", type=int, default=696)
        sub.add_argument("--checkpoint-dir", default="model_checkpoints")
        sub.add_argument("--tag", default="v1")
        sub.add_argument("--episodes", type=int, default=400)
        sub.add_argument("--max-steps", type=int, default=50)
        sub.add_argument("--gamma", type=float, default=0.95)
        sub.add_argument("--learning-rate", type=float, default=3e-4)
        sub.add_argument("--log-every", type=int, default=5)
        sub.add_argument("--evaluate-episodes", type=int, default=10)
        sub.add_argument("--visualize", action="store_true")
        sub.add_argument("--no-progress", action="store_true")

    parser.set_defaults(command="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_single_task(args)
    elif args.command == "full":
        run_full_pipeline(args)
    elif args.command == "see-envs":
        run_see_envs(args)
    else:
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
