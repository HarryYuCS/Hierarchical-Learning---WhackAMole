import gymnasium as gym
import numpy as np

def create_env(seed : int):
    env = gym.make("FetchReach-v4")

    # TODO 1 : specify the properties of the env

    return env

def sample_goal():
    """
    Choose the goal from a list of discrete positions representing thoe holes where the moles pop up from
    """
    holes = [
        np.array([0.70, 0.80, 0.30]),
        np.array([1.35, 0.80, 0.30]),
        np.array([2.00, 0.80, 0.30]),
        np.array([0.70, 1.00, 0.30]),
        np.array([1.35, 1.00, 0.30]),
        np.array([2.00, 1.00, 0.30]),
        np.array([0.70, 1.20, 0.30]),
        np.array([1.35, 1.20, 0.30]),
        np.array([2.00, 1.20, 0.30]),
    ]

def visualize(env: gym.Env, algorithm=None, video_name="test"):
    """
        Visualize a policy network for a given algorithm on a single episode

        Args:
            - env_name: Name of the gym environment to roll out `algorithm` in,
                it will be instantiated using gym.make or make_vec_env.
            - algorithm (PPOActor): Algorithm whose policy network will be rolled
                out for the episode. If no algorithm is passed in, a random policy
                will be visualized.
            - video_name (str): Name for the mp4 file of the episode that will be
                saved (omit .mp4). Only used when running on local machine.
    """

    def get_action(obs):
        if not algorithm:
            return env.action_space.sample()
        else:
            return algorithm.select_action(obs)

    if USING_COLAB:
        import renderlab as rl

        directory = './video'
        env = rl.RenderFrame(env, "output/")
        obs, info = env.reset()

        for i in range(500):
            action = get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated: break
        env.play()

    else:
        import cv2

        video = cv2.VideoWriter(f"{video_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24, (600,400))
        obs = env.reset()

        for i in range(500):
            action = get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated: break

            im = env.render(mode='rgb_array')
            im = im[:,:,::-1]
            video.write(im)

        video.release()
        env.close()
        print(f"Video saved as {video_name}.mp4")