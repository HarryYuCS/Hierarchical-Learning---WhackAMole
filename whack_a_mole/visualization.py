import gymnasium as gym
import numpy as np

def visualize_no_actor():
    env = gym.make("FetchReach-v4", render_mode="human")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # Replace with your agent's policy
        observation, reward, terminated, truncated, info = env.step(action)
        
        # render() is called automatically if render_mode="human" is set, 
        # but you can call it manually if needed.
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

def visualize(env: gym.Env, algorithm=None, video_name="test"):
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

    for i in range(500):
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: break

        im = env.render()
        if im is not None:
            video.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    video.release()
    env.close()
    print(f"Video saved as {video_name}.mp4")
