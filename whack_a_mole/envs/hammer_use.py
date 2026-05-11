from __future__ import annotations

import mujoco
import numpy as np

from whack_a_mole.envs.pickup import PickupEnv


class HammerUseEnv(PickupEnv):
    """Hammer-use task initialized from a stable pre-grasped hammer state."""

    debug_reset = False
    pregrasp_offset = np.array([0.0, 0.0, 0.012], dtype=np.float64)
    target_hold_aperture = 0.01
    strike_speed_threshold = 0.2
    min_downward_velocity = 0.08
    downward_alignment_threshold = 0.65
    hit_radius = 0.08
    strike_ready_radius = 0.10
    strike_height_tolerance = 0.08
    aim_height = 0.12
    hit_bonus = 10.0

    def __init__(self, reward_type: str = "dense", render_mode: str | None = "human", **kwargs):
        super().__init__(
            reward_type=reward_type,
            render_mode=render_mode,
            **kwargs,
        )

    def _place_hammer_in_grasp(self) -> None:
        """Place hammer into a plausible in-gripper pose."""
        grip_pos = self.get_gripper_position()
        if not np.isfinite(grip_pos).all() or float(np.linalg.norm(grip_pos)) < 0.5:
            grip_pos = np.array([1.35, 0.75, 0.50], dtype=np.float64)
        hammer_qpos = self._utils.get_joint_qpos(self.model, self.data, "hammer:joint")
        hammer_qpos[:3] = grip_pos + self.pregrasp_offset
        hammer_qpos[3:] = self.get_hammer_orientation_quat()
        self._utils.set_joint_qpos(self.model, self.data, "hammer:joint", hammer_qpos)
        hammer_qvel = self._utils.get_joint_qvel(self.model, self.data, "hammer:joint")
        hammer_qvel[:] = 0.0
        self._utils.set_joint_qvel(self.model, self.data, "hammer:joint", hammer_qvel)
        mujoco.mj_forward(self.model, self.data)

    def _close_gripper(self) -> None:
        """Close gripper around the placed hammer handle."""
        self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", self.target_hold_aperture)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", self.target_hold_aperture)
        if self.data.ctrl.shape[0] >= 2:
            self.data.ctrl[-2:] = self.target_hold_aperture
        mujoco.mj_forward(self.model, self.data)

    def _force_gripper_close_action(self, action) -> np.ndarray:
        """Bias gripper action toward stable hold without over-squeezing."""
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] >= 4:
            gripper_aperture = float(np.mean(self.get_gripper_state()[0]))
            if gripper_aperture > self.target_hold_aperture + 0.002:
                action[3] = min(float(action[3]), -0.4)
            else:
                action[3] = 0.0
        return action

    def _is_sane_pose(self) -> bool:
        """Check that gripper/hammer pose is finite and inside workspace."""
        handle = self.get_hammer_handle_position()
        tip = self.get_hammer_tip_position()
        grip = self.get_gripper_position()
        all_finite = np.isfinite(handle).all() and np.isfinite(tip).all() and np.isfinite(grip).all()
        if not all_finite:
            return False
        handle_in_bounds = (
            0.6 < handle[0] < 2.2
            and 0.25 < handle[1] < 1.35
            and 0.2 < handle[2] < 1.0
        )
        grip_in_bounds = (
            0.6 < grip[0] < 2.2
            and 0.25 < grip[1] < 1.35
            and 0.2 < grip[2] < 1.2
        )
        grip_to_handle = float(np.linalg.norm(grip - handle))
        fingers_closed = bool(np.mean(self.get_gripper_state()[0]) < 0.02)
        return bool(handle_in_bounds and grip_in_bounds and grip_to_handle < 0.08 and fingers_closed)

    def reset(self, *args, **kwargs):
        """Reset from stable pickup state, then enforce pre-grasp setup."""
        obs = None
        info = {}

        for _ in range(3):
            obs, info = super().reset(*args, **kwargs)
            self._place_hammer_in_grasp()
            self._close_gripper()
            if self._is_sane_pose():
                break
        else:
            raise RuntimeError("HammerUseEnv reset failed to produce sane pre-grasp state")

        obs = self._get_obs()
        obs["desired_goal"] = self.goal.copy()
        obs["achieved_goal"] = self.get_hammer_tip_position()
        info["hammer_lifted"] = True
        grasped = self.is_hammer_grasped()
        info["hammer_grasped"] = grasped
        info["hammer_held"] = bool(grasped)
        info["phase"] = "hammer_use"

        if self.debug_reset:
            print(
                "HammerUse reset:",
                "grip=", self.get_gripper_position(),
                "handle=", self.get_hammer_handle_position(),
                "tip=", self.get_hammer_tip_position(),
            )
        return obs, info

    def compute_dense_reward(self, achieved_goal, goal, info=None):
        horizontal_distance = float(self.check_horizontal_distance(achieved_goal, goal))
        vertical_offset = float(achieved_goal[2] - goal[2])
        aim_height_error = abs(vertical_offset - self.aim_height)
        near_for_strike = horizontal_distance < self.strike_ready_radius

        reward = -self.step_penalty - 5.0 * horizontal_distance
        if near_for_strike:
            reward -= 2.0 * aim_height_error

        if info is not None:
            downward_speed = max(0.0, -float(info.get("hammer_tip_velocity_z", 0.0)))
            above_mole = vertical_offset > 0.0
            if near_for_strike and above_mole:
                reward += 3.0 * downward_speed

            near_mole = horizontal_distance < self.hit_radius
            hovering = near_mole and (abs(vertical_offset) < self.strike_height_tolerance)
            if hovering and downward_speed < self.min_downward_velocity:
                reward -= 1.0

        return float(reward)

    def compute_reward(self, achieved_goal, goal, info=None):
        return float(self.compute_dense_reward(achieved_goal, goal, info))

    def step(self, action):
        action = self._force_gripper_close_action(action)
        obs, _, terminated, truncated, info = super().step(action)
        obs["achieved_goal"] = self.get_hammer_tip_position()
        achieved = obs["achieved_goal"]
        goal = obs["desired_goal"]

        velocity = self.get_hammer_tip_velocity()
        speed = float(np.linalg.norm(velocity))
        downward_alignment = 0.0 if speed < 1e-8 else float(-velocity[2] / speed)
        distance = float(self.check_distance(achieved, goal))
        horizontal_distance = float(self.check_horizontal_distance(achieved, goal))
        height_error = float(abs(achieved[2] - goal[2]))

        is_hit = bool(
            horizontal_distance < self.hit_radius
            and height_error < self.strike_height_tolerance
            and speed >= self.strike_speed_threshold
            and float(velocity[2]) <= -self.min_downward_velocity
            and downward_alignment >= self.downward_alignment_threshold
        )

        reward = float(self.compute_reward(achieved, goal, {"hammer_tip_velocity_z": float(velocity[2])}))
        if is_hit:
            reward += self.hit_bonus

        info["is_success"] = is_hit
        info["strike_valid"] = is_hit
        info["phase"] = "hammer_use"
        info["hammer_lifted"] = True
        grasped = self.is_hammer_grasped()
        info["hammer_grasped"] = grasped
        info["hammer_held"] = bool(grasped)
        info["hammer_tip_distance"] = float(distance)
        info["hammer_tip_horizontal_distance"] = float(horizontal_distance)
        info["hammer_tip_height_error"] = float(height_error)
        info["hammer_tip_speed"] = float(speed)
        info["hammer_tip_velocity_z"] = float(velocity[2])
        info["hammer_tip_downward_alignment"] = float(downward_alignment)
        info["hammer_use_reward"] = float(reward)
        info["hammer_use_action_xyz_norm"] = float(np.linalg.norm(action[:3]))
        if action.shape[0] >= 4:
            info["hammer_use_action_grip"] = float(action[3])

        if is_hit:
            self.goal = self._sample_goal()
            self._move_goal_marker()
            obs["desired_goal"] = self.goal.copy()
        return obs, reward, terminated, truncated, info
