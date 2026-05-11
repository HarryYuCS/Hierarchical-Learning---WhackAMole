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

    def annotate(frame, step_idx, reward, info):
        if not show_overlay:
            return frame
        import cv2

        out = frame.copy()
        lines = [
            f"step={step_idx}",
            f"reward={reward:.3f}",
            f"phase={info.get('phase', 'n/a')}",
            f"pickup_phase={info.get('pickup_phase', 'n/a')}",
            f"held={int(bool(info.get('hammer_held', False)))} grasped={int(bool(info.get('hammer_grasped', False)))} lifted={int(bool(info.get('hammer_lifted', False)))}",
            f"dist_handle={info.get('grip_to_hammer_handle', float('nan')):.3f}",
            f"success={int(bool(info.get('is_success', False)))}",
        ]
        y = 24
        for line in lines:
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22
        return out

    for i in range(max_steps):
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: break

        im = env.render()
        if im is not None:
            im = annotate(im, i + 1, float(reward), info)
            video.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    video.release()
    env.close()
    print(f"Video saved as {video_name}.mp4")
