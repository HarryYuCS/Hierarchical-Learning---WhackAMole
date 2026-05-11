from __future__ import annotations

import numpy as np

from whack_a_mole.envs.hammer_use import HammerUseEnv
from whack_a_mole.envs.pickup import PickupEnv


class EndToEndEnv(PickupEnv):
    pickup_goal_tracks_handle = False

    def compute_dense_reward(self, achieved_goal, goal, info=None):
        pickup_reward = super().compute_dense_reward(achieved_goal, goal, info)
        lifted = self.is_hammer_lifted()
        if not lifted:
            return pickup_reward

        hammer_reward = HammerUseEnv.compute_dense_reward(self, achieved_goal, goal, info)
        return pickup_reward + hammer_reward

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        achieved = obs["achieved_goal"]
        goal = obs["desired_goal"]
        velocity = self.get_hammer_tip_velocity()
        is_hit, distance, horizontal_distance, height_error, speed, downward_alignment = self.is_valid_strike(achieved, goal)
        reward = float(self.compute_dense_reward(achieved, goal, {"hammer_tip_velocity_z": float(velocity[2])}))
        if is_hit and self.is_hammer_lifted():
            reward += self.hit_bonus
            self.goal = self._sample_goal()
            self._move_goal_marker()
            obs["desired_goal"] = self.goal.copy()

        info["is_success"] = bool(is_hit and self.is_hammer_lifted())
        info["phase"] = "hammer_use" if self.is_hammer_lifted() else "pickup"
        info["hammer_tip_distance"] = float(distance)
        info["hammer_tip_horizontal_distance"] = float(horizontal_distance)
        info["hammer_tip_height_error"] = float(height_error)
        info["hammer_tip_speed"] = float(speed)
        info["hammer_tip_velocity_z"] = float(velocity[2])
        info["hammer_tip_downward_alignment"] = float(downward_alignment)
        return obs, reward, terminated, truncated, info
