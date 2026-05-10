import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium_robotics.envs.fetch.reach import MujocoFetchEnv
import numpy as np
import os
import mujoco

class WhackAMoleEnv(MujocoFetchEnv):
    hit_bonus = 10.0
    strike_speed_threshold = 0.2
    min_downward_velocity = 0.08
    downward_alignment_threshold = 0.65
    hit_radius = 0.08
    strike_ready_radius = 0.10
    strike_height_tolerance = 0.08
    aim_height = 0.12
    step_penalty = 0.05

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

    def check_horizontal_distance(self, achieved_goal, goal):
        return np.linalg.norm(achieved_goal[..., :2]-goal[..., :2], axis=-1)

    def get_hammer_tip_position(self):
        return self.data.site('hammer_tip').xpos.copy()

    def get_hammer_tip_velocity(self):
        site_id = self.model.site('hammer_tip').id
        site_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            site_vel,
            0,
        )
        return site_vel[3:].copy()

    def _get_obs(self):
        obs = super()._get_obs()
        obs["achieved_goal"] = self.get_hammer_tip_position()
        return obs

    def _move_goal_marker(self):
        target_site_id = self.model.site("target0").id
        target_body_id = int(self.model.site("target0").bodyid[0])
        target_body_pos = self.data.body(target_body_id).xpos
        self.model.site_pos[target_site_id] = self.goal - target_body_pos
        mujoco.mj_forward(self.model, self.data)

    def _render_callback(self):
        self._move_goal_marker()

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._move_goal_marker()
        obs["achieved_goal"] = self.get_hammer_tip_position()
        obs["desired_goal"] = self.goal.copy()
        return obs, info

    def _sample_goal(self):
        """
        Choose the goal from a list of discrete positions representing the holes where moles pop up.
        """
        holes = [
            np.array([1.25, 0.55, 0.45]),
            np.array([1.45, 0.55, 0.45]),
            np.array([1.65, 0.55, 0.45]),
            np.array([1.25, 0.75, 0.45]),
            np.array([1.45, 0.75, 0.45]),
            np.array([1.65, 0.75, 0.45]),
            np.array([1.25, 0.95, 0.45]),
            np.array([1.45, 0.95, 0.45]),
            np.array([1.65, 0.95, 0.45]),
        ]
        return holes[int(self.np_random.integers(len(holes)))].copy()

    def is_valid_strike(self, achieved_goal, goal):
        velocity = self.get_hammer_tip_velocity()
        speed = np.linalg.norm(velocity)
        downward_alignment = 0.0 if speed < 1e-8 else -velocity[2] / speed
        distance = self.check_distance(achieved_goal, goal)
        horizontal_distance = self.check_horizontal_distance(achieved_goal, goal)
        height_error = abs(achieved_goal[2] - goal[2])
        is_hit = bool(
            horizontal_distance < self.hit_radius
            and height_error < self.strike_height_tolerance
            and speed >= self.strike_speed_threshold
            and velocity[2] <= -self.min_downward_velocity
            and downward_alignment >= self.downward_alignment_threshold
        )
        return is_hit, distance, horizontal_distance, height_error, speed, downward_alignment

    def compute_dense_reward(self, achieved_goal, goal, info=None):
        horizontal_distance = self.check_horizontal_distance(achieved_goal, goal)
        vertical_offset = achieved_goal[..., 2] - goal[..., 2]
        aim_height_error = np.abs(vertical_offset - self.aim_height)
        near_for_strike = horizontal_distance < self.strike_ready_radius

        reward = -self.step_penalty - 5.0 * horizontal_distance
        reward -= np.asarray(near_for_strike, dtype=np.float32) * 0.5 * aim_height_error

        if info is not None:
            downward_speed = max(0.0, -float(info.get("hammer_tip_velocity_z", 0.0)))
            near_mole = horizontal_distance < self.hit_radius
            strike_ready = horizontal_distance < self.strike_ready_radius
            above_mole = vertical_offset > 0.0
            strike_gate = np.asarray(strike_ready, dtype=np.float32) * np.asarray(above_mole, dtype=np.float32)
            reward += strike_gate * (3.0 * downward_speed)

            hovering = near_mole & (np.abs(vertical_offset) < self.strike_height_tolerance)
            if downward_speed < self.min_downward_velocity:
                reward -= np.asarray(hovering, dtype=np.float32)

            ready_but_not_striking = strike_ready & above_mole
            if downward_speed < self.min_downward_velocity:
                reward -= 0.25 * np.asarray(ready_but_not_striking, dtype=np.float32)

        return reward
 
    def compute_reward(self, achieved_goal, goal, info=None):
        '''
            computes the actor's reward based on how environment was initialized
            sparse - binary reward hit bonus for valid strikes, 0 elsewhere
            dense - reward aims above the mole and only favors contact when striking downward
        '''
        if self.reward_type == "sparse":
            if info is not None and "valid_strike" in info:
                return self.hit_bonus * np.asarray(info["valid_strike"], dtype=np.float32)
            return np.zeros_like(self.check_horizontal_distance(achieved_goal, goal), dtype=np.float32)
        else:
            return self.compute_dense_reward(achieved_goal, goal, info)

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        obs['achieved_goal'] = self.get_hammer_tip_position()
        achieved = obs['achieved_goal']
        goal = obs['desired_goal']

        velocity = self.get_hammer_tip_velocity()
        (
            is_hit,
            distance,
            horizontal_distance,
            height_error,
            speed,
            downward_alignment,
        ) = self.is_valid_strike(achieved, goal)
        reward_info = {
            "valid_strike": is_hit,
            "hammer_tip_velocity_z": float(velocity[2]),
        }
        reward = float(self.compute_reward(achieved, goal, reward_info))
        if self.reward_type == "dense" and is_hit:
            reward += self.hit_bonus

        info['is_success'] = is_hit
        info['hammer_tip_distance'] = float(distance)
        info['hammer_tip_horizontal_distance'] = float(horizontal_distance)
        info['hammer_tip_hit_radius'] = float(self.hit_radius)
        info['hammer_tip_height_error'] = float(height_error)
        info['hammer_tip_speed'] = float(speed)
        info['hammer_tip_velocity_z'] = float(velocity[2])
        info['hammer_tip_downward_alignment'] = float(downward_alignment)

        if is_hit:
            self.goal = self._sample_goal()
            self._move_goal_marker()
            obs['desired_goal'] = self.goal.copy()
            
        return obs, reward, terminated, truncated, info

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
        return self.unwrapped.get_hammer_tip_velocity()

    def observation(self, obs):
        hammer_pos = self.unwrapped.get_hammer_tip_position()
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
    env = WhackAMoleEnv(render_mode=render_mode, reward_type=reward_type)
    env = WhackAMoleObservationWrapper(env)
    return env
