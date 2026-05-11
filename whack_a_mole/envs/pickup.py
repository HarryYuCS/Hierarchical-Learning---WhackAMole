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
    travel_scale = 22.0
    close_radius = 0.045
    close_reward = 0.0
    close_miss_penalty = 0.6
    max_aperture = 0.05
    handle_reach_bonus = 20.0
    pickup_goal_tracks_handle = True
    assist_grasp_closure = False
    early_close_action_penalty = 3.0
    early_closed_penalty = 8.0
    close_action_reward = 4.0
    closed_gripper_reward = 10.0
    hover_height = 0.03
    grasp_center_height = 0.08
    hover_xy_radius = 0.045
    hover_height_tolerance = 0.08
    reach_radius = 0.035
    reach_dwell_steps = 8
    close_success_dwell_steps = 5
    reach_height_tolerance = 0.04
    phase_target_bonus = 16.0
    descent_ready_radius = 0.06
    min_descent_velocity = 0.02
    ideal_descent_velocity = 0.08
    max_descent_velocity = 0.18
    descent_velocity_reward = 8.0
    fast_descent_penalty = 25.0
    unaligned_descent_penalty = 12.0
    lateral_velocity_penalty = 3.0
    head_clearance_radius = 0.10
    head_collision_penalty = 3.0
    head_avoid_radius = 0.14
    head_avoid_penalty = 12.0
    head_approach_penalty = 8.0
    active_goal_progress_scale = 35.0
    horizontal_goal_scale = 12.0
    vertical_goal_scale = 8.0
    low_while_misaligned_penalty = 12.0
    hammer_disturbance_penalty = 25.0
    open_gripper_reward = 2.0
    phase_bonus = {
        "hover": 35.0,
        "close": 15.0,
        "reach": 30.0,
    }

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
        self._reset_handle_xy = hammer_qpos[:2].copy()
        hammer_qpos[3:] = self.get_hammer_orientation_quat()
        self._utils.set_joint_qpos(self.model, self.data, "hammer:joint", hammer_qpos)
        mujoco.mj_forward(self.model, self.data)
        self._prev_grip_to_handle = None
        self._prev_grip_to_active_goal = None
        self._pickup_stage = "hover"
        self._pickup_reached_hover = False
        self._pickup_reach_dwell = 0
        self._pickup_close_dwell = 0
        self._earned_phase_bonuses = set()
        return did_reset

    def _get_handle_axis(self):
        """Return the horizontal handle-to-head direction."""
        handle_pos = self.get_hammer_handle_position()
        tip_pos = self.get_hammer_tip_position()
        axis = tip_pos - handle_pos
        axis[2] = 0.0
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis / norm

    def _get_hover_position(self):
        """Return a target directly above the gripper-frame grasp target."""
        hover_pos = self._get_grasp_center_position()
        hover_pos[2] += self.hover_height
        return hover_pos

    def _get_grasp_center_position(self):
        """Return the gripper-frame target that places open fingers around the handle."""
        handle_pos = self.get_hammer_handle_position()
        grasp_pos = handle_pos.copy()
        grasp_pos[2] += self.grasp_center_height
        return grasp_pos

    def _get_descend_position(self):
        """Return the open-gripper target just above the handle."""
        return self._get_grasp_center_position()

    def get_hammer_head_position(self):
        """Return world position of the hammer head body."""
        return self.data.body("hammer_head").xpos.copy()

    def get_gripper_velocity(self):
        """Return gripper site linear velocity in world coordinates."""
        site_id = self.model.site("robot0:grip").id
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

    def get_pickup_stage_features(self):
        """Return one-hot pickup stage features for observations."""
        stage = getattr(self, "_pickup_stage", "approach")
        return np.array(
            [
                float(stage == "hover"),
                float(stage == "close"),
                float(stage == "reach"),
                float(stage == "lift"),
            ],
            dtype=np.float32,
        )

    def get_head_side_amount(self):
        """Positive when the gripper is on the head side of the handle."""
        grip_pos = self.get_gripper_position()
        handle_pos = self.get_hammer_handle_position()
        handle_axis = self._get_handle_axis()
        return max(0.0, float(np.dot(grip_pos - handle_pos, handle_axis)))

    def _get_active_pickup_goal(self):
        """Return the active pickup target for the current phase."""
        stage = getattr(self, "_pickup_stage", "hover")
        if stage in {"close", "reach"}:
            return self._get_descend_position()
        return self._get_hover_position()

    def _update_pickup_stage(self):
        grip_pos = self.get_gripper_position()
        hover_pos = self._get_hover_position()
        hover_dist = float(np.linalg.norm(grip_pos - hover_pos))

        if hover_dist < self.reach_radius:
            self._pickup_reach_dwell = getattr(self, "_pickup_reach_dwell", 0) + 1
        else:
            self._pickup_reach_dwell = 0

        if getattr(self, "_pickup_reached_hover", False) or self._pickup_reach_dwell >= self.reach_dwell_steps:
            self._pickup_reached_hover = True
            descend_dist = float(np.linalg.norm(grip_pos - self._get_descend_position()))
            if descend_dist < self.reach_radius and self._is_gripper_closed():
                self._pickup_close_dwell = getattr(self, "_pickup_close_dwell", 0) + 1
            else:
                self._pickup_close_dwell = 0
            self._pickup_stage = "reach" if self._pickup_close_dwell >= self.close_success_dwell_steps else "close"
        else:
            self._pickup_close_dwell = 0
            self._pickup_stage = "hover"

    def _consume_phase_bonus(self):
        """Return one-time reward for newly completed pickup milestones."""
        grip_pos = self.get_gripper_position()
        hover_pos = self._get_hover_position()
        dist_hover = float(np.linalg.norm(grip_pos - hover_pos))
        stage = getattr(self, "_pickup_stage", "hover")
        earned = getattr(self, "_earned_phase_bonuses", set())

        bonus = 0.0
        if "hover" not in earned and self._pickup_reach_dwell >= self.reach_dwell_steps:
            bonus += self.phase_bonus["hover"]
            earned.add("hover")
        if "close" not in earned and stage in {"close", "reach"}:
            bonus += self.phase_bonus["close"]
            earned.add("close")
        if "reach" not in earned and stage == "reach":
            bonus += self.phase_bonus["reach"]
            earned.add("reach")

        self._earned_phase_bonuses = earned
        return bonus

    def _sync_pickup_goal(self, obs):
        """Expose the current pickup waypoint as the active goal for pickup training."""
        if not self.pickup_goal_tracks_handle:
            return obs
        goal_pos = self._get_active_pickup_goal()
        grip_pos = self.get_gripper_position()
        self.goal = goal_pos.copy()
        self._move_goal_marker()
        obs["achieved_goal"] = grip_pos.copy()
        obs["desired_goal"] = goal_pos.copy()
        return obs

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._pickup_stage = "hover"
        self._pickup_reached_hover = False
        self._pickup_reach_dwell = 0
        self._pickup_close_dwell = 0
        self._earned_phase_bonuses = set()
        self._update_pickup_stage()
        return self._sync_pickup_goal(obs), info

    def is_hammer_lifted(self):
        """Return whether hammer handle is above lift threshold."""
        return bool(self.get_hammer_handle_position()[2] > self.lifted_height)

    def _is_gripper_closed(self):
        gripper_state, _ = self.get_gripper_state()
        return bool(np.mean(gripper_state) < self.grasp_close_threshold)

    def is_hammer_grasped(self):
        """Return whether the gripper is closed near the hammer handle."""
        stage = getattr(self, "_pickup_stage", "hover")
        if stage in {"hover", "descend", "reach"}:
            return False
        grip_pos = self.get_gripper_position()
        handle_pos = self.get_hammer_handle_position()
        grip_to_handle = np.linalg.norm(grip_pos - handle_pos)
        grasp_radius = self.handle_grasp_radius if stage == "lift" else self.close_radius
        return bool(grip_to_handle < grasp_radius and self._is_gripper_closed())

    def compute_dense_reward(self, achieved_goal, goal, info=None):
        """Compute dense shaping reward for top-down open-gripper reaching.

        Args:
            achieved_goal: Unused for pickup reward.
            goal: Active waypoint for the current pickup phase.
            info: Optional info dict.

        Returns:
            Scalar dense reward encouraging open-gripper top-down reach.
        """
        del achieved_goal
        info = info or {}
        
        # 1. Coordinate Mapping
        grip_pos = self.get_gripper_position()
        handle_pos = self.get_hammer_handle_position()
        active_goal = np.asarray(goal, dtype=np.float64)
        head_pos = self.get_hammer_head_position()
        head_side_amount = self.get_head_side_amount()
        
        # Distances
        active_goal_dist = float(np.linalg.norm(grip_pos - active_goal))
        active_goal_xy_dist = float(np.linalg.norm((grip_pos - active_goal)[:2]))
        active_goal_z_dist = abs(float(grip_pos[2] - active_goal[2]))
        vertical_offset = float(grip_pos[2] - active_goal[2])
        head_dist = float(np.linalg.norm(grip_pos - head_pos))
        head_vector = head_pos - grip_pos
        handle_xy_shift = float(np.linalg.norm(handle_pos[:2] - self._reset_handle_xy))
        handle_lift = max(0.0, float(handle_pos[2] - self.hammer_start[2] - 0.02))
        gripper_velocity = np.asarray(info.get("gripper_velocity", self.get_gripper_velocity()), dtype=np.float64)
        downward_speed = max(0.0, -float(gripper_velocity[2]))
        lateral_speed = float(np.linalg.norm(gripper_velocity[:2]))
        head_direction = head_vector / max(float(np.linalg.norm(head_vector)), 1e-6)
        head_approach_speed = max(0.0, float(np.dot(gripper_velocity, head_direction)))
        
        # Gripper Stats
        gripper_qpos, _ = self.get_gripper_state()
        gripper_aperture = float(np.mean(gripper_qpos))
        closure_amount = np.clip(1.0 - (gripper_aperture / self.max_aperture), 0, 1)
        close_command = float(info.get("gripper_close_command", 0.0))
        
        # Logical States
        stage = getattr(self, "_pickup_stage", "hover")
        horizontal_dist = float(np.linalg.norm((grip_pos - handle_pos)[:2]))
        gripper_open_amount = np.clip(gripper_aperture / self.max_aperture, 0, 1)

        # 2. Top-down reach reward.
        reward = -float(self.step_penalty)
        reward -= self.travel_scale * active_goal_dist
        reward -= self.horizontal_goal_scale * active_goal_xy_dist
        reward -= self.vertical_goal_scale * active_goal_z_dist
        reward -= 4.0 * horizontal_dist
        reward -= self.hammer_disturbance_penalty * (handle_xy_shift + handle_lift)
        reward -= self.head_avoid_penalty * max(0.0, (self.head_avoid_radius - head_dist) / self.head_avoid_radius)
        reward -= self.head_approach_penalty * head_approach_speed
        reward -= self.head_collision_penalty * max(0.0, head_side_amount / self.head_clearance_radius)
        reward += self.handle_reach_bonus * np.exp(-20.0 * active_goal_dist)
        if stage in {"hover", "close"}:
            reward += self.phase_target_bonus * np.exp(-30.0 * active_goal_dist)
            above_goal = vertical_offset > 0.0
            ready_for_descent = active_goal_xy_dist < self.descent_ready_radius and above_goal
            if ready_for_descent:
                speed_error = abs(downward_speed - self.ideal_descent_velocity)
                reward += self.descent_velocity_reward * np.exp(-25.0 * speed_error)
                reward -= self.fast_descent_penalty * max(0.0, downward_speed - self.max_descent_velocity)
                if downward_speed < self.min_descent_velocity and vertical_offset > self.reach_height_tolerance:
                    reward -= 0.5
            else:
                reward -= self.unaligned_descent_penalty * downward_speed
            reward -= self.lateral_velocity_penalty * lateral_speed
            if horizontal_dist > self.hover_xy_radius:
                safe_height = handle_pos[2] + self.hover_height * 0.5
                reward -= self.low_while_misaligned_penalty * max(0.0, safe_height - grip_pos[2])
            reward -= self.head_collision_penalty * max(0.0, (self.head_clearance_radius - head_dist) / self.head_clearance_radius)
        elif stage == "reach":
            reward += self.phase_target_bonus
        
        # Potential Shaping (Delta reward for moving closer)
        if self._prev_grip_to_active_goal is not None:
            reward += self.active_goal_progress_scale * (self._prev_grip_to_active_goal - active_goal_dist)

        # 3. Keep the gripper open until the hover pose is stable, then learn to close.
        if stage == "hover":
            reward += self.open_gripper_reward * gripper_open_amount
            reward -= self.early_closed_penalty * closure_amount
            reward -= self.early_close_action_penalty * close_command
        else:
            near_descend_target = active_goal_dist < self.reach_radius
            reward += self.close_action_reward * close_command * float(near_descend_target)
            reward += self.closed_gripper_reward * closure_amount * float(near_descend_target)
            if not near_descend_target:
                reward -= self.early_closed_penalty * closure_amount
                reward -= self.early_close_action_penalty * close_command
            reward -= 4.0 * max(0.0, active_goal_dist - self.reach_radius)

        self._last_pickup_phase = stage
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
        action = np.asarray(action, dtype=np.float32).copy()
        self._update_pickup_stage()
        grip_to_handle_before = float(np.linalg.norm(self.get_gripper_position() - self.get_hammer_handle_position()))
        if self.assist_grasp_closure and action.shape[0] >= 4 and grip_to_handle_before < self.close_radius:
            action[3] = min(float(action[3]), -0.7)

        obs, _, terminated, truncated, info = super().step(action)
        self._update_pickup_stage()
        obs = self._sync_pickup_goal(obs)
        reward_info = {
            "gripper_close_command": max(0.0, -float(action[3])) if action.shape[0] >= 4 else 0.0,
            "gripper_velocity": self.get_gripper_velocity(),
        }
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], reward_info))
        phase_bonus = self._consume_phase_bonus()
        reward += phase_bonus
        lifted = self.is_hammer_lifted()
        grasped = self.is_hammer_grasped()
        grip_to_handle = float(np.linalg.norm(self.get_gripper_position() - self.get_hammer_handle_position()))
        grip_to_active_goal = float(np.linalg.norm(self.get_gripper_position() - self._get_active_pickup_goal()))
        self._prev_grip_to_handle = grip_to_handle
        self._prev_grip_to_active_goal = grip_to_active_goal
        gripper_qpos, _ = self.get_gripper_state()
        gripper_velocity = self.get_gripper_velocity()
        reached_closed = bool(self._pickup_stage == "reach")
        if reached_closed:
            terminated = True
        info["is_success"] = reached_closed
        info["phase"] = "pickup"
        info["hammer_lifted"] = lifted
        info["hammer_grasped"] = grasped
        info["hammer_held"] = bool(grasped or lifted)
        info["handle_reached_open"] = bool(self._pickup_reached_hover)
        info["handle_reached_closed"] = reached_closed
        info["grip_to_hammer_handle"] = grip_to_handle
        info["grip_to_pickup_goal"] = grip_to_active_goal
        info["grip_to_grasp_center"] = float(np.linalg.norm(self.get_gripper_position() - self._get_grasp_center_position()))
        info["grip_to_hover"] = float(np.linalg.norm(self.get_gripper_position() - self._get_hover_position()))
        info["grip_to_descend"] = float(np.linalg.norm(self.get_gripper_position() - self._get_descend_position()))
        info["grip_to_hammer_head"] = float(np.linalg.norm(self.get_gripper_position() - self.get_hammer_head_position()))
        info["grip_to_hammer_tip"] = float(np.linalg.norm(self.get_gripper_position() - self.get_hammer_tip_position()))
        info["head_side_amount"] = self.get_head_side_amount()
        info["phase_bonus"] = float(phase_bonus)
        info["earned_phase_bonuses"] = tuple(sorted(self._earned_phase_bonuses))
        info["reach_dwell"] = int(self._pickup_reach_dwell)
        info["close_dwell"] = int(self._pickup_close_dwell)
        info["gripper_aperture"] = float(np.mean(gripper_qpos))
        info["gripper_velocity_z"] = float(gripper_velocity[2])
        info["gripper_lateral_speed"] = float(np.linalg.norm(gripper_velocity[:2]))
        head_vector = self.get_hammer_head_position() - self.get_gripper_position()
        head_direction = head_vector / max(float(np.linalg.norm(head_vector)), 1e-6)
        info["head_approach_speed"] = float(max(0.0, np.dot(gripper_velocity, head_direction)))
        info["gripper_can_close"] = bool(action.shape[0] >= 4)
        info["grasp_assist_active"] = bool(self.assist_grasp_closure and grip_to_handle_before < self.close_radius)
        info["pickup_phase"] = getattr(self, "_last_pickup_phase", "travel")
        # debug print to check gripper aperature distance
        # gripper_qpos, _ = self.get_gripper_state()
        # print(f"DEBUG: Gripper qpos: {gripper_qpos} | Mean: {np.mean(gripper_qpos)}")
        return obs, reward, terminated, truncated, info
