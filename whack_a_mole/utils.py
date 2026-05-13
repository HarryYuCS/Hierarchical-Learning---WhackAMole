import torch
import random
import numpy as np
import gymnasium as gym
from pathlib import Path

from whack_a_mole.actors import SACActor, TrainConfig
from whack_a_mole.envs import create_env
from whack_a_mole.training_viz import plot_training_metrics

def reseed(seed, env=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if env is not None:
        env.unwrapped._np_random = gym.utils.seeding.np_random(seed)[0]


def build_train_config(args) -> TrainConfig:
    return TrainConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        seed=args.seed,
        show_progress=not args.no_progress,
        log_every=args.log_every,
    )


def default_ckpt_name(task: str) -> str:
    return f"sac_{task}.zip"


def load_actor_from_checkpoint(*, task: str, seed: int, checkpoint_path: str):
    env = create_env(render_mode=None, task=task)
    reseed(seed, env)
    actor = SACActor.load(checkpoint_path, env=env)
    print(f"Loaded checkpoint for {task} from {checkpoint_path}")
    env.close()
    return actor


def load_or_train_actor(
    *,
    task: str,
    seed: int,
    ckpt_dir: Path,
    ckpt_name: str,
    train_config: TrainConfig,
    plot_train_metrics: bool,
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
    if plot_train_metrics:
        plot_path = ckpt_dir / f"{ckpt_path.stem}_metrics.png"
        plot_training_metrics(result, plot_path, title=f"SAC {task} training metrics")
        print(f"Saved training plot to {plot_path}")
    print(f"Trained {task}: timesteps={result.timesteps}")
    env.close()
    return actor
