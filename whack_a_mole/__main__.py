from pathlib import Path

from whack_a_mole.utils import reseed
from whack_a_mole.create_env import create_env
from whack_a_mole.visualization import visualize

from whack_a_mole.algorithms import PPOActor, TrainConfig, make_algorithm


def main():
    seed = 696
    train_env = create_env(render_mode=None, reward_type="dense")
    reseed(seed, train_env)

    # TODO 2
    # Initialize and train the actors
    train_config = TrainConfig(episodes=250,
                               max_steps_per_episode=200,
                               gamma=0.95,
                               learning_rate=3e-4,
                               seed=seed,
                               show_progress=True,
                               log_every=5)
    actor = make_algorithm(name="ppo", environment=train_env)

    ckpt_path = Path("model_checkpoints/ppo_dense.zip")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists():
        actor = PPOActor.load(str(ckpt_path), env=train_env)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        train_result = actor.train(train_env, train_config)
        actor.save(str(ckpt_path))
        print(
            f"Training complete: episodes={train_config.episodes} timesteps={train_result.timesteps}"
        )

    # TODO 3
    # Evaluate,
    # Visualize performance + videos
    train_env.close()
    video_env = create_env(render_mode="rgb_array")
    reseed(seed, video_env)
    visualize(video_env, actor, "ppo_test_dense")

if __name__ == "__main__":
    main()
