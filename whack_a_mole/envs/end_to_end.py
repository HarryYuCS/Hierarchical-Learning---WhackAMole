from __future__ import annotations

import numpy as np

from whack_a_mole.envs.hammer_use import HammerUseEnv
from whack_a_mole.envs.pickup import PickupEnv


class EndToEndEnv(PickupEnv):
    hit_bonus = 10.0
    strike_speed_threshold = 0.2
    min_downward_velocity = 0.08
    downward_alignment_threshold = 0.65
    hit_radius = 0.08
    strike_ready_radius = 0.10
    strike_height_tolerance = 0.08
    aim_height = 0.12
    drop_penalty = 5.0
    target_hold_aperture = 0.01

    def _in_hammer_use_phase(self) -> bool:
        """Return whether hammer-use rewards/phase should be active."""
        return bool(self.is_hammer_grasped() or self.is_hammer_lifted())

    def _force_gripper_close_action(self, action) -> np.ndarray:
        """Bias gripper action toward stable hold during hammer-use phase."""
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape[0] >= 4:
            gripper_aperture = float(np.mean(self.get_gripper_state()[0]))
            if gripper_aperture > self.target_hold_aperture + 0.002:
                action[3] = min(float(action[3]), -0.4)
            else:
                action[3] = 0.0
        return action

    def compute_dense_reward(self, achieved_goal, goal, info=None):
        pickup_reward = super().compute_dense_reward(achieved_goal, goal, info)
        if not self._in_hammer_use_phase():
            return pickup_reward

        hammer_reward = HammerUseEnv.compute_dense_reward(self, achieved_goal, goal, info)
        return pickup_reward + hammer_reward

    def step(self, action):
        was_in_hammer_use_phase = self._in_hammer_use_phase()
        if was_in_hammer_use_phase:
            action = self._force_gripper_close_action(action)
        obs, _, terminated, truncated, info = super().step(action)
        achieved = obs["achieved_goal"]
        goal = obs["desired_goal"]
        velocity = self.get_hammer_tip_velocity()
        is_hit, distance, horizontal_distance, height_error, speed, downward_alignment = self.is_valid_strike(achieved, goal)
        reward = float(self.compute_dense_reward(achieved, goal, {"hammer_tip_velocity_z": float(velocity[2])}))
        in_hammer_use_phase = self._in_hammer_use_phase()
        dropped_hammer = bool(was_in_hammer_use_phase and not in_hammer_use_phase)
        if dropped_hammer:
            reward -= self.drop_penalty
        if is_hit and in_hammer_use_phase:
            reward += self.hit_bonus
            self.goal = self._sample_goal()
            self._move_goal_marker()
            obs["desired_goal"] = self.goal.copy()

        info["is_success"] = bool(is_hit and in_hammer_use_phase)
        info["phase"] = "hammer_use" if in_hammer_use_phase else "pickup"
        info["strike_valid"] = bool(is_hit and in_hammer_use_phase)
        info["hammer_dropped"] = dropped_hammer
        info["drop_penalty"] = float(-self.drop_penalty if dropped_hammer else 0.0)
        info["hammer_tip_distance"] = float(distance)
        info["hammer_tip_horizontal_distance"] = float(horizontal_distance)
        info["hammer_tip_height_error"] = float(height_error)
        info["hammer_tip_speed"] = float(speed)
        info["hammer_tip_velocity_z"] = float(velocity[2])
        info["hammer_tip_downward_alignment"] = float(downward_alignment)
        return obs, reward, terminated, truncated, info
