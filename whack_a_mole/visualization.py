import gymnasium as gym
import numpy as np

def visualize_no_actor(env :gym.Env, video_name="test"):
    """
        Visualize an environment to see if it looks right

    """

    import cv2
    obs, _ = env.reset()

    first_frame = env.render()
    if first_frame is None:
        raise ValueError("visualize requires env with render_mode='rgb_array' to save mp4")
    if not isinstance(first_frame, np.ndarray) or first_frame.ndim != 3:
        raise ValueError("env.render() did not return an RGB frame")

    height, width = first_frame.shape[:2]
    video = cv2.VideoWriter(
        f"{video_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        24,
        (width, height),
    )
    video.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    for i in range(500):
        no_action = np.zeros(env.action_space.shape)
        no_action[-1] = 1
        obs, reward, terminated, truncated, info = env.step(no_action)
        if terminated or truncated: break

        im = env.render()
        if im is not None:
            video.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    video.release()
    env.close()
    print(f"Video saved as {video_name}.mp4")


def visualize(env: gym.Env, algorithm=None, video_name="test", show_overlay: bool = False, max_steps: int = 500):
    """
        Visualize a policy network for a given algorithm on a single episode

        Args:
            - env_name: Name of the gym environment to roll out `algorithm` in,
                it will be instantiated using gym.make or make_vec_env.
            - algorithm (RLAlgorithm): Algorithm whose policy network will be rolled
                out for the episode. If no algorithm is passed in, a random policy
                will be visualized.
            - video_name (str): Name for the mp4 file of the episode that will be
                saved (omit .mp4). Only used when running on local machine.
            - show_overlay (bool): If True, render debug text on each frame.
            - max_steps (int): Maximum rollout steps for the video.
    """

    def get_action(obs):
        if not algorithm:
            return env.action_space.sample()
        else:
            return algorithm.predict(obs, deterministic=True)

    import cv2
    obs, _ = env.reset()

    first_frame = env.render()
    if first_frame is None:
        raise ValueError("visualize requires env with render_mode='rgb_array' to save mp4")
    if not isinstance(first_frame, np.ndarray) or first_frame.ndim != 3:
        raise ValueError("env.render() did not return an RGB frame")

    height, width = first_frame.shape[:2]
    video = cv2.VideoWriter(
        f"{video_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        24,
        (width, height),
    )
    video.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    def _fmt(value, digits=3):
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "n/a"

    def annotate(frame, step_idx, reward, cumulative_reward, info, action):
        if not show_overlay:
            return frame
        import cv2

        out = frame.copy()
        action = np.asarray(action, dtype=np.float32).ravel()
        action_xyz_norm = float(np.linalg.norm(action[:3])) if action.size >= 3 else float("nan")
        action_grip = float(action[3]) if action.size >= 4 else float("nan")

        lines = [
            f"step={step_idx}",
            f"reward={reward:.3f} cum_reward={cumulative_reward:.3f}",
            f"phase={info.get('phase', 'n/a')}",
            f"pickup_phase={info.get('pickup_phase', 'n/a')}",
            f"held={int(bool(info.get('hammer_held', False)))} grasped={int(bool(info.get('hammer_grasped', False)))} lifted={int(bool(info.get('hammer_lifted', False)))}",
            f"dist_handle={_fmt(info.get('grip_to_hammer_handle', float('nan')))} aperture={_fmt(info.get('gripper_aperture', float('nan')))}",
            f"tip_dist={_fmt(info.get('hammer_tip_distance', float('nan')))} tip_xy={_fmt(info.get('hammer_tip_horizontal_distance', float('nan')))} tip_vz={_fmt(info.get('hammer_tip_velocity_z', float('nan')))}",
            f"tip_speed={_fmt(info.get('hammer_tip_speed', float('nan')))} align={_fmt(info.get('hammer_tip_downward_alignment', float('nan')))}",
            f"act_xyz_norm={_fmt(action_xyz_norm)} act_grip={_fmt(action_grip)}",
            f"actor_act_xyz_norm={_fmt(info.get('hammer_use_action_xyz_norm', float('nan')))} actor_act_grip={_fmt(info.get('hammer_use_action_grip', float('nan')))}",
            f"success={int(bool(info.get('is_success', False)))}",
        ]
        y = 24
        for line in lines:
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22
        return out

    cumulative_reward = 0.0
    for i in range(max_steps):
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        if terminated or truncated: break

        im = env.render()
        if im is not None:
            im = annotate(im, i + 1, float(reward), cumulative_reward, info, action)
            video.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    video.release()
    env.close()
    print(f"Video saved as {video_name}.mp4")
