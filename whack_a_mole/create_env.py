import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium_robotics.envs.fetch.reach import MujocoFetchEnv
import numpy as np
import random
import os
import mujoco

class WhackAMoleEnv(MujocoFetchEnv):
    def __init__(self, reward_type="sparse", render_mode='human', **kwargs):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "assets", "fetch", "reach.xml")
        
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
            gripper_extra_height=0.0,
            block_gripper=True,
            has_object=False,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            **kwargs
        )
        
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
        
        if self.reward_type == "sparse":
            return (distance < self.distance_threshold).astype(np.float32)
        else:
            return -distance

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Use self.unwrapped to ensure we are talking to the MuJoCo engine
        model = self.unwrapped.model
        data = self.unwrapped.data
        
        site_id = model.site('hammer_tip').id
        site_vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, site_vel, 0)
        
        strike_velocity = site_vel[4] 
        
        achieved = obs['achieved_goal']
        goal = obs['desired_goal']
        distance = self.check_distance(achieved, goal)
        
        if distance < self.distance_threshold and strike_velocity < -0.2:
            self.goal = self._sample_goal()
            obs['desired_goal'] = self.goal.copy()
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
            np.array([0.70, 0.5451, 0.45]),
            np.array([1.20, 0.5451, 0.45]),
            np.array([1.70, 0.5451, 0.45]),
            np.array([0.70, 0.7451, 0.45]),
            np.array([1.20, 0.7451, 0.45]),
            np.array([1.70, 0.7451, 0.45]),
            np.array([0.70, 0.9451, 0.45]),
            np.array([1.20, 0.9451, 0.45]),
            np.array([1.70, 0.9451, 0.45]),
        ]
        return random.choice(holes).copy()

class WhackAMoleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_shape = (12,)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-np.inf, np.inf, new_shape, dtype=np.float32),
            "achieved_goal": env.observation_space["achieved_goal"],
            "desired_goal": env.observation_space["desired_goal"],
        })

    def get_hammer_velocity(self):
        # The key is self.unwrapped
        model = self.unwrapped.model
        data = self.unwrapped.data
        
        site_id = model.site('hammer_tip').id
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, vel, 0)
        
        return vel[3:] 

    def observation(self, obs):
        hammer_pos = obs['achieved_goal']
        goal_pos = obs['desired_goal']
        hammer_vel = self.get_hammer_velocity()
        rel_pos = goal_pos - hammer_pos
        
        low_dim_obs = np.concatenate([
            hammer_pos, 
            hammer_vel, 
            goal_pos, 
            rel_pos
        ]).astype(np.float32)
        
        return {
            "observation": low_dim_obs,
            "achieved_goal": hammer_pos,
            "desired_goal": goal_pos
        }

def create_env(render_mode=None, reward_type="sparse"):
    env = WhackAMoleEnv(render_mode=render_mode)
    env = WhackAMoleObservationWrapper(env, reward_type=reward_type)
    return env
