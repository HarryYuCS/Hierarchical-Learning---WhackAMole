from __future__ import annotations

import mujoco
import numpy as np

from whack_a_mole.envs.base import BaseWhackEnv


class PickupEnv(BaseWhackEnv):
    """Pickup-only task where the policy learns to grasp and lift the hammer.

    Attributes:
        hammer_start: Nominal hammer handle start position.
        hammer_start_noise: Uniform random noise applied at reset.
        handle_grasp_radius: Distance threshold considered near handle.
        lifted_height: Handle z-threshold indicating successful lift.
    """
    hammer_start = np.array([1.30, 0.80, 0.48])
    hammer_start_noise = np.array([0.02, 0.02, 0.0])
    handle_grasp_radius = 0.055
    lifted_height = 0.50
    grasp_close_threshold = 0.025
    pickup_bonus = 8.0
    travel_scale = 3.0
    close_radius = 0.045
    close_reward = 3.0
    close_miss_penalty = 0.6
    max_aperture = 0.05
    handle_reach_bonus = 3.0
    pickup_goal_tracks_handle = True

    def __init__(self, reward_type: str = "dense", render_mode: str | None = "human", **kwargs):
        """Initialize pickup environment with free hammer and active gripper.

        Args:
            reward_type: Reward mode.
            render_mode: Gymnasium render mode.
            **kwargs: Extra kwargs for the base environment.
        """
        super().__init__(
            reward_type=reward_type,
            render_mode=render_mode,
            xml_file="hammer_pickup.xml",
            block_gripper=False,
            gripper_extra_height=0.15,
            initial_qpos={
                "robot0:slide0": 0.403,
                "robot0:slide1": 0.481,
                "robot0:slide2": 0.12,
                "hammer:joint": [1.25, 0.55, 0.48, 0.5, 0.5, -0.5, -0.5],
            },
            **kwargs,
        )

    def _reset_sim(self):
        """Reset simulation and randomize hammer start pose.

        Returns:
            True if reset succeeded.
        """
        did_reset = super()._reset_sim()
        hammer_qpos = self._utils.get_joint_qpos(self.model, self.data, "hammer:joint")
        hammer_qpos[:3] = self.hammer_start + self.np_random.uniform(-self.hammer_start_noise, self.hammer_start_noise)
        self._reset_handle_xy = hammer_qpos[:2].copy()
        hammer_qpos[3:] = self.get_hammer_orientation_quat()
        self._utils.set_joint_qpos(self.model, self.data, "hammer:joint", hammer_qpos)
        mujoco.mj_forward(self.model, self.data)
        self._prev_grip_to_handle = None
        return did_reset

    def _sync_pickup_goal(self, obs):
        """Expose the handle as the active goal for pickup training."""
        if not self.pickup_goal_tracks_handle:
            return obs
        handle_pos = self.get_hammer_handle_position()
        grip_pos = self.get_gripper_position()
        self.goal = handle_pos.copy()
        self._move_goal_marker()
        obs["achieved_goal"] = grip_pos.copy()
        obs["desired_goal"] = handle_pos.copy()
        return obs

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return self._sync_pickup_goal(obs), info

    def is_hammer_lifted(self):
        """Return whether hammer handle is above lift threshold."""
        return bool(self.get_hammer_handle_position()[2] > self.lifted_height)

    def is_hammer_grasped(self):
        """Return whether the gripper is closed near the hammer handle."""
        grip_pos = self.get_gripper_position()
        handle_pos = self.get_hammer_handle_position()
        gripper_state, _ = self.get_gripper_state()
        grip_to_handle = np.linalg.norm(grip_pos - handle_pos)
        gripper_closed = float(np.mean(gripper_state) < self.grasp_close_threshold)
        near_handle = float(grip_to_handle < self.handle_grasp_radius)
        return bool(near_handle * gripper_closed > 0.5)

    def compute_dense_reward(self, achieved_goal, goal, info=None):
        """Compute dense shaping reward for pickup behavior.

        Args:
            achieved_goal: Unused for pickup reward.
            goal: Unused for pickup reward.
            info: Optional info dict.

        Returns:
            Scalar dense reward encouraging reach, grasp, and lift.
        """
        del achieved_goal, goal, info
        
        # 1. Coordinate Mapping
        grip_pos = self.get_gripper_position()
        handle_pos = self.get_hammer_handle_position()
        
        # Distances
        dist = float(np.linalg.norm(grip_pos - handle_pos))
        height_dist = abs(grip_pos[2] - handle_pos[2])
        
        # Gripper Stats
        gripper_qpos, _ = self.get_gripper_state()
        gripper_aperture = float(np.mean(gripper_qpos))
        closure_amount = np.clip(1.0 - (gripper_aperture / self.max_aperture), 0, 1)
        
        # Logical States
        grasped = float(self.is_hammer_grasped())
        near_handle = float(dist < self.handle_grasp_radius)

        # 2. Base Travel Reward (The "Funnel" to the hammer)
        reward = -float(self.step_penalty)
        reward -= self.travel_scale * dist
        reward -= 2.0 * height_dist  # Explicit motivation to stop hovering
        reward += self.handle_reach_bonus * np.exp(-20.0 * dist)
        if dist < self.close_radius:
            reward += self.close_reward * (1.0 - dist / self.close_radius)
        
        # Potential Shaping (Delta reward for moving closer)
        if self._prev_grip_to_handle is not None:
            reward += 5.0 * (self._prev_grip_to_handle - dist)
        self._prev_grip_to_handle = dist

        # 3. Grasping Gradient
        if near_handle > 0.5:
            # Reward closing ONLY when near the handle
            nearness_multiplier = np.exp(-15.0 * dist)
            reward += 8.0 * nearness_multiplier * closure_amount
        else:
            # Small penalty for closing fingers in empty air
            reward -= 0.5 * closure_amount

        # 4. Lifting Reward
        if grasped > 0.5:
            reward += self.pickup_bonus
            # Scale lift reward by height achieved
            lift_progress = max(0.0, handle_pos[2] - self.hammer_start[2])
            reward += 25.0 * lift_progress 
            
            if self.is_hammer_lifted():
                reward += 10.0 # Success kicker

        self._last_pickup_phase = "lift" if grasped > 0.5 else "travel"
        return reward

    def compute_reward(self, achieved_goal, goal, info=None):
        """Project reward entrypoint for the pickup task.

        Args:
            achieved_goal: Current achieved goal.
            goal: Current desired goal.
            info: Optional info dict.

        Returns:
            Dense reward value.
        """
        return self.compute_dense_reward(achieved_goal, goal, info)

    def step(self, action):
        """Step environment and annotate pickup-specific info fields.

        Args:
            action: Action applied to the environment.

        Returns:
            Tuple of ``(obs, reward, terminated, truncated, info)``.
        """
        obs, _, terminated, truncated, info = super().step(action)
        obs = self._sync_pickup_goal(obs)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"]))
        lifted = self.is_hammer_lifted()
        grasped = self.is_hammer_grasped()
        grip_to_handle = float(np.linalg.norm(self.get_gripper_position() - self.get_hammer_handle_position()))
        gripper_qpos, _ = self.get_gripper_state()
        info["is_success"] = lifted
        info["phase"] = "pickup"
        info["hammer_lifted"] = lifted
        info["hammer_grasped"] = grasped
        info["hammer_held"] = bool(grasped or lifted)
        info["grip_to_hammer_handle"] = grip_to_handle
        info["gripper_aperture"] = float(np.mean(gripper_qpos))
        info["gripper_can_close"] = True
        info["pickup_phase"] = getattr(self, "_last_pickup_phase", "travel")
        # debug print to check gripper aperature distance
        # gripper_qpos, _ = self.get_gripper_state()
        # print(f"DEBUG: Gripper qpos: {gripper_qpos} | Mean: {np.mean(gripper_qpos)}")
        return obs, reward, terminated, truncated, info
