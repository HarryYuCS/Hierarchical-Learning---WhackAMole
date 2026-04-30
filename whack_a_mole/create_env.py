import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
import numpy as np
import random
import os

class WhackAMoleEnv(MujocoFetchReachEnv):
    def __init__(self, reward_type="sparse", *args, **kwargs):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        xml_path = os.path.join(current_dir, "assets", "fetch", "reach.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Missing XML at: {xml_path}")

        self.reward_type = reward_type
        
        kwargs.pop('model_path', None)
        super().__init__(xml_path, **kwargs)
        
        # Verification prints
        print(f"--- Environment Loaded ---")
        print(f"XML Path: {xml_path}")
        print(f"Table Size: {self.model.geom('table0').size}")
        print(f"Total Geoms: {self.model.ngeom}")

    def check_distance(self, achieved_goal, goal):
        return np.linalg.norm(achieved_goal-goal, axis=-1)
 
    def compute_reward(self, achieved_goal, goal, info=None):
        '''
            computes the actor's reward based on how environment was initialized
            sparse - binary reward 1 at goal 0 elsewhere
            dense - reward is negative euclidean distance from goal
        '''
        distance = self.check_distance(achieved_goal, goal)
        if self.reward_type=="sparse":
            return (distance < self.distance_threshold).astype(np.float32)
        else:
            return -distance

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        achieved = obs['achieved_goal']
        goal = obs['desired_goal']
        distance = self.check_distance(achieved, goal)
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        if distance < self.distance_threshold:
            self.goal = self._sample_goal()
            obs['desired_goal'] = self.goal
            # bonus for reaching the goal
            # TODO:
            # requires: velocity within a magnitude and direction threshold to achieeve
            reward += 10
            info['is_success'] = True
        else:
            info['is_success'] = False
        return obs, reward, terminated, truncated, info

    def _sample_goal(self):
        """
        Choose the goal from a list of discrete positions representing thoe holes where the moles pop up from
        """
        holes = [
            np.array([0.70, 0.80, 0.45]),
            np.array([1.35, 0.80, 0.45]),
            np.array([2.00, 0.80, 0.45]),
            np.array([0.70, 1.00, 0.45]),
            np.array([1.35, 1.00, 0.45]),
            np.array([2.00, 1.00, 0.45]),
            np.array([0.70, 1.20, 0.45]),
            np.array([1.35, 1.20, 0.45]),
            np.array([2.00, 1.20, 0.45]),
        ]
        return random.choice(holes).copy()

gym.register(
    id='WhackAMoleFetch',
    entry_point=WhackAMoleEnv,
    kwargs={}
)

def create_env():
    return gym.make('WhackAMoleFetch', render_mode='human')