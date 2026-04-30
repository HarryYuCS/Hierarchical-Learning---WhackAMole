import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium_robotics.envs.fetch.reach import MujocoFetchEnv
import numpy as np
import random
import os


class WhackAMoleEnv(MujocoFetchEnv):
    def __init__(self, reward_type="sparse", render_mode='human', **kwargs):
        # 1. Setup path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "assets", "fetch", "reach.xml")
        
        # 2. Call the base constructor with ALL required positional arguments
        # These values recreate the 'Reach' environment configuration
        super().__init__(
            model_path=xml_path,
            n_substeps=20,
            initial_qpos={
                "robot0:slide0": 0.403,
                "robot0:slide1": 0.481,
                "robot0:slide2": 0.0,
            },
            reward_type=reward_type,
            render_mode=render_mode,
            # The 8 missing required arguments:
            gripper_extra_height=0.0,
            block_gripper=True,
            has_object=False,          # Set to False since we are reaching, not picking
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            **kwargs
        )
        
        # Verification
        if 'table0' in [self.model.geom(i).name for i in range(self.model.ngeom)]:
            print(f"SUCCESS: Custom XML Loaded. Table size: {self.model.geom('table0').size}")

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


# gym.register(
#     id='WhackAMoleFetch',
#     entry_point=WhackAMoleEnv,
#     kwargs={}
# )

def create_env():
    return WhackAMoleEnv(render_mode='human')