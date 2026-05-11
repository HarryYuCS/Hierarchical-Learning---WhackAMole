from pathlib import Path

from whack_a_mole.utils import reseed
from whack_a_mole.create_env import create_env
from whack_a_mole.visualization import visualize, visualize_no_actor

from whack_a_mole.actors import PPOActor, StitchedPPOActor, TrainConfig, SACActor


def evaluate_actor(env, actor, episodes: int = 10):
    rewards = []
    successes = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        final_success = 0.0
        for _ in range(500):
            action = actor.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            final_success = float(info.get("is_success", 0.0))
            if terminated or truncated:
                break
        rewards.append(total)
        successes.append(final_success)
    return float(sum(rewards) / max(len(rewards), 1)), float(sum(successes) / max(len(successes), 1))


def inspect_stitched_switching(env, stitched_actor, episodes: int = 3):
    pickup_steps = 0
    hammer_use_steps = 0
    switch_events = 0

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
        switch_events = stitched_actor.switch_count

    print(
        f"Stitched policy usage: pickup_steps={pickup_steps}, "
        f"hammer_use_steps={hammer_use_steps}, switches={switch_events}"
    )


def main():
    seed = 696
    checkpoint_version = "v3"
    train_config = TrainConfig(
        episodes=100,
        max_steps_per_episode=200,
        gamma=0.95,
        learning_rate=3e-4,
        seed=seed,
        show_progress=True,
        log_every=5,
    )
    ckpt_dir = Path("model_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def load_or_train(task: str, ckpt_name: str) -> PPOActor:
        env = create_env(render_mode=None, task=task)
        reseed(seed, env)
        actor = PPOActor(environment=env)
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            actor = PPOActor.load(str(ckpt_path), env=env)
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            result = actor.train(env, train_config)
            actor.save(str(ckpt_path))
            print(f"Trained {task}: timesteps={result.timesteps}")
        env.close()
        return actor

    pickup_actor = load_or_train("pickup", f"ppo_dense_pickup_{checkpoint_version}.zip")
    hammer_use_actor = load_or_train("hammer_use", f"ppo_dense_hammer_use_{checkpoint_version}.zip")
    e2e_actor = load_or_train("end_to_end", f"ppo_dense_end_to_end_{checkpoint_version}.zip")

    stitched_actor = StitchedPPOActor(pickup_actor=pickup_actor, hammer_use_actor=hammer_use_actor)

    eval_env = create_env(render_mode=None, task="end_to_end")
    reseed(seed, eval_env)
    inspect_stitched_switching(eval_env, stitched_actor, episodes=3)
    stitched_mean_reward, stitched_success = evaluate_actor(eval_env, stitched_actor, episodes=10)
    e2e_eval = e2e_actor.evaluate(eval_env, episodes=10)
    eval_env.close()

    print(
        f"Stitched success={stitched_success:.3f}, mean_reward={stitched_mean_reward:.2f}; "
        f"End-to-end success={e2e_eval.success_rate:.3f}, mean_reward={e2e_eval.mean_reward:.2f}"
    )

    pickup_video_env = create_env(render_mode="rgb_array", task="pickup")
    reseed(seed, pickup_video_env)
    visualize(pickup_video_env, pickup_actor, "ppo_pickup_only", show_overlay=True)

    hammer_video_env = create_env(render_mode="rgb_array", task="hammer_use")
    reseed(seed, hammer_video_env)
    visualize(hammer_video_env, hammer_use_actor, "ppo_hammer_use_only", show_overlay=True)

    stitched_video_env = create_env(render_mode="rgb_array", task="end_to_end")
    reseed(seed, stitched_video_env)
    visualize(stitched_video_env, stitched_actor, "ppo_stitched_end_to_end", show_overlay=True)

    e2e_video_env = create_env(render_mode="rgb_array", task="end_to_end")
    reseed(seed, e2e_video_env)
    visualize(e2e_video_env, e2e_actor, "ppo_e2e_end_to_end", show_overlay=True)

def see_envs():
    # visualize_no_actor(create_env(render_mode="rgb_array", task="end_to_end"), video_name="e2e_env")
    visualize_no_actor(create_env(render_mode="rgb_array", task="pickup"), video_name="pickup_env")
    # visualize_no_actor(create_env(render_mode="rgb_array", task="hammer_use"), video_name="hammer_use_env")

def pickup_only():
    seed = 696
    train_config = TrainConfig(
        episodes=150,
        max_steps_per_episode=100,
        gamma=0.95,
        learning_rate=3e-4,
        seed=seed,
        show_progress=True,
        log_every=5,
    )
    ckpt_dir = Path("model_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def load_or_train(task: str, ckpt_name: str) -> SACActor:
        env = create_env(render_mode=None, task=task)
        reseed(seed, env)
        actor = SACActor(environment=env)
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            actor = SACActor.load(str(ckpt_path), env=env)
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            result = actor.train(env, train_config)
            actor.save(str(ckpt_path))
            print(f"Trained {task}: timesteps={result.timesteps}")
        env.close()
        return actor

    pickup_actor = load_or_train("pickup", "sac_dense_pickup_head_avoid_descend_close.zip")

    pickup_video_env = create_env(render_mode="rgb_array", task="pickup")
    reseed(seed, pickup_video_env)
    visualize(pickup_video_env, pickup_actor, "sac_pickup_only", show_overlay=True)

if __name__ == "__main__":
    # main()
    # see_envs()
    pickup_only()
