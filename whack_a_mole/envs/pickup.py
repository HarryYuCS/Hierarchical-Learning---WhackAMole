from __future__ import annotations

import mujoco
import numpy as np

from whack_a_mole.envs.base import BaseWhackEnv


class PickupEnv(BaseWhackEnv):
    """Pickup-only task where the policy learns to reach the hammer handle.

    Attributes:
        hammer_start: Nominal hammer handle start position.
        hammer_start_noise: Uniform random noise applied at reset.
        handle_grasp_radius: Distance threshold considered near handle.
        lifted_height: Handle z-threshold used by downstream hammer-use handoff checks.
    """
    hammer_start = np.array([1.30, 0.80, 0.46])
    hammer_start_noise = np.array([0.02, 0.02, 0.0])
    handle_grasp_radius = 0.08
    lifted_height = 0.50
    grasp_close_threshold = 0.05
    pickup_bonus = 0.0
    close_radius = 0.045
    max_aperture = 0.05
    assist_grasp_closure = False

    pickup_aim_height = 0.12
    pickup_ready_radius = 0.05
    pickup_height_tolerance = 0.03

    pickup_hover_height = 0.30

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
                "hammer:joint": [1.25, 0.55, 0.46, 0.5, 0.5, -0.5, -0.5],
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
        hammer_qpos[3:] = self.get_hammer_orientation_quat()
        self._utils.set_joint_qpos(self.model, self.data, "hammer:joint", hammer_qpos)
        mujoco.mj_forward(self.model, self.data)
        return did_reset

    def get_hammer_head_position(self):
        """Return world position of the hammer head body."""
        return self.data.body("hammer_head").xpos.copy()

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def is_hammer_lifted(self):
        """Return whether hammer handle is above lift threshold."""
        return bool(self.get_hammer_handle_position()[2] > self.lifted_height)

    def _is_gripper_closed(self):
        gripper_state, _ = self.get_gripper_state()
        return bool(np.mean(gripper_state) < self.grasp_close_threshold)

    def is_hammer_grasped(self):
        """Return whether the gripper is closed near the hammer handle."""
        grip_pos = self.get_gripper_position()
        handle_pos = self.get_hammer_handle_position()
        grip_to_handle = np.linalg.norm(grip_pos - handle_pos)
        return bool(grip_to_handle < self.close_radius and self._is_gripper_closed())

    def compute_dense_reward(self, achieved_goal, goal, info=None):
        """Compute simplified dense shaping reward for pickup.

        Args:
            achieved_goal: Unused for pickup reward.
            goal: Unused for pickup reward.
            info: Optional info dict.

        Returns:
            Scalar dense reward encouraging handle alignment and gripping.
        """
        del achieved_goal, goal

        grip_pos = self.get_gripper_position()
        handle_pos = self.get_hammer_handle_position()

        horizontal_distance = float(self.check_horizontal_distance(grip_pos, handle_pos))
        vertical_offset = float(grip_pos[2] - handle_pos[2])
        near_for_grasp = horizontal_distance < self.pickup_ready_radius

        reward = -self.step_penalty - 5.0 * horizontal_distance
        if near_for_grasp:
            aim_height_error = abs(vertical_offset - self.pickup_aim_height)
            reward -= 2.0 * aim_height_error
        # change : require hover to enforce approach from above
        else:
            hover_height_error = abs(vertical_offset - self.pickup_hover_height)
            reward -= 2.0 * hover_height_error

        gripper_qpos, _ = self.get_gripper_state()
        gripper_aperture = float(np.mean(gripper_qpos))
        grip_engagement = float(np.clip(1.0 - (gripper_aperture / self.max_aperture), 0.0, 1.0))

        if info is not None:
            aimed_handle = near_for_grasp and (abs(vertical_offset) < self.pickup_height_tolerance)
            if aimed_handle:
                reward += 3.0 * grip_engagement

        grasped = self.is_hammer_grasped()
        lifted = self.is_hammer_lifted()
        if grasped:
            reward += self.pickup_bonus
        if lifted:
            reward += 10.0

        return float(reward)

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
        action = np.asarray(action, dtype=np.float32).copy()
        grip_to_handle_before = float(np.linalg.norm(self.get_gripper_position() - self.get_hammer_handle_position()))
        if self.assist_grasp_closure and action.shape[0] >= 4 and grip_to_handle_before < self.close_radius:
            action[3] = min(float(action[3]), -0.7)

        obs, _, terminated, truncated, info = super().step(action)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))
        lifted = self.is_hammer_lifted()
        grasped = self.is_hammer_grasped()
        grip_to_handle = float(np.linalg.norm(self.get_gripper_position() - self.get_hammer_handle_position()))
        gripper_qpos, _ = self.get_gripper_state()
        info["is_success"] = bool(grasped)
        info["phase"] = "pickup"
        info["hammer_lifted"] = lifted
        info["hammer_grasped"] = grasped
        info["hammer_held"] = bool(grasped or lifted)
        info["grip_to_hammer_handle"] = grip_to_handle
        info["gripper_aperture"] = float(np.mean(gripper_qpos))
        info["gripper_can_close"] = bool(action.shape[0] >= 4)
        info["grasp_assist_active"] = bool(self.assist_grasp_closure and grip_to_handle_before < self.close_radius)
        return obs, reward, terminated, truncated, info
