from __future__ import annotations

import gymnasium as gym
import numpy as np


class HammerUseObservationWrapper(gym.ObservationWrapper):
    """Observation wrapper for hammer-use policy inputs (12 features)."""

    def __init__(self, env):
        """Initialize hammer-use observation wrapper.

        Args:
            env: Wrapped Gymnasium environment.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
                "achieved_goal": env.observation_space["achieved_goal"],
                "desired_goal": env.observation_space["desired_goal"],
            }
        )

    def observation(self, obs):
        """Build low-dimensional hammer-use observation dictionary.

        Args:
            obs: Original observation dict.

        Returns:
            Dict observation with 12D policy features.
        """
        hammer_pos = self.unwrapped.get_hammer_tip_position()
        goal_pos = obs["desired_goal"]
        hammer_vel = self.unwrapped.get_hammer_tip_velocity()
        grip_pos = self.unwrapped.get_gripper_position()
        gripper_state, _ = self.unwrapped.get_gripper_state()
        gripper_aperture = np.array([float(np.mean(gripper_state))], dtype=np.float32)
        held_flag = np.array([float(self.unwrapped.is_hammer_grasped())], dtype=np.float32)
        rel_pos = goal_pos - hammer_pos
        low_dim_obs = np.concatenate(
            [hammer_pos, hammer_vel, goal_pos, rel_pos, grip_pos, gripper_aperture, held_flag]
        ).astype(np.float32)
        return {
            "observation": low_dim_obs,
            "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32),
            "desired_goal": goal_pos,
        }


class PickupObservationWrapper(gym.ObservationWrapper):
    """Observation wrapper for pickup/end-to-end policy inputs (38 features)."""

    def __init__(self, env):
        """Initialize pickup observation wrapper.

        Args:
            env: Wrapped Gymnasium environment.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-np.inf, np.inf, (38,), dtype=np.float32),
                "achieved_goal": env.observation_space["achieved_goal"],
                "desired_goal": env.observation_space["desired_goal"],
            }
        )

    def observation(self, obs):
        """Build low-dimensional pickup/end-to-end observation dictionary.

        Args:
            obs: Original observation dict.

        Returns:
            Dict observation with 38D policy features.
        """
        grip_pos = self.unwrapped.get_gripper_position()
        gripper_state, _ = self.unwrapped.get_gripper_state()
        handle_pos = self.unwrapped.get_hammer_handle_position()
        head_pos = self.unwrapped.get_hammer_head_position()
        hammer_pos = self.unwrapped.get_hammer_tip_position()
        hammer_vel = self.unwrapped.get_hammer_tip_velocity()
        goal_pos = obs["desired_goal"]
        grip_to_goal = goal_pos - grip_pos
        grip_to_handle = handle_pos - grip_pos
        grip_to_head = head_pos - grip_pos
        tip_to_goal = goal_pos - hammer_pos
        near_handle = np.array(
            [
                np.linalg.norm(grip_to_handle) < self.unwrapped.handle_grasp_radius,
                float(self.unwrapped.is_hammer_grasped() or self.unwrapped.is_hammer_lifted()),
            ],
            dtype=np.float32,
        )
        low_dim_obs = np.concatenate(
            [
                grip_pos,
                gripper_state,
                handle_pos,
                hammer_pos,
                hammer_vel,
                goal_pos,
                grip_to_goal,
                grip_to_handle,
                head_pos,
                grip_to_head,
                tip_to_goal,
                self.unwrapped.get_pickup_stage_features(),
                near_handle,
            ]
        ).astype(np.float32)
        return {
            "observation": low_dim_obs,
            "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32),
            "desired_goal": goal_pos,
        }


def hammer_use_obs_from_full_obs(obs: dict) -> dict:
    """Adapt full pickup/end-to-end observation to hammer-use feature format.

    Args:
        obs: Observation dict containing 25D ``observation`` field.

    Returns:
        Observation dict compatible with hammer-use policy wrapper schema.
    """
    full = np.asarray(obs["observation"], dtype=np.float32)
    grip_pos = full[0:3]
    gripper_state = full[3:5]
    hammer_pos = full[8:11]
    hammer_vel = full[11:14]
    goal_pos = full[14:17]
    held_flag = np.array([full[24]], dtype=np.float32)
    gripper_aperture = np.array([float(np.mean(gripper_state))], dtype=np.float32)
    rel_pos = goal_pos - hammer_pos
    low_dim = np.concatenate(
        [
            hammer_pos,
            hammer_vel,
            goal_pos,
            rel_pos,
            grip_pos,
            gripper_aperture,
            held_flag,
        ]
    ).astype(np.float32)
    return {
        "observation": low_dim,
        "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32),
        "desired_goal": np.asarray(obs["desired_goal"], dtype=np.float32),
    }
