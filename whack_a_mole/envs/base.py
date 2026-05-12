from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium_robotics.envs.fetch.reach import MujocoFetchEnv


class BaseWhackEnv(MujocoFetchEnv):
    """Shared MuJoCo environment utilities for whack-a-mole tasks.

    This base class centralizes common robot/hammer state queries, goal marker
    placement, and strike validation logic used by pickup, hammer-use, and
    end-to-end task variants.

    Attributes:
        step_penalty: Per-step shaping penalty.
    """
    step_penalty = 0.05
    # Keep hammer horizontal and yaw it so the handle faces the robot while
    # the head points away.
    hammer_quat = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)
    # mujoco.mju_mulQuat(hammer_quat, hammer_quat, np.array([0.0, 0.0, 1.0, 0.0]))

    def __init__(
        self,
        reward_type: str = "dense",
        render_mode: str | None = "human",
        xml_file: str = "reach.xml",
        block_gripper: bool = True,
        gripper_extra_height: float = 0.0,
        initial_qpos: dict | None = None,
        **kwargs,
    ):
        """Initialize the base MuJoCo whack-a-mole environment.

        Args:
            reward_type: Reward mode configured in the underlying Fetch env.
            render_mode: Gymnasium render mode.
            xml_file: XML model filename under ``assets/fetch``.
            block_gripper: Whether to lock gripper fingers.
            gripper_extra_height: Extra gripper height offset.
            initial_qpos: Optional initial joint positions.
            **kwargs: Extra kwargs passed to ``MujocoFetchEnv``.
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "..", "assets", "fetch", xml_file)
        initial_qpos = initial_qpos or {
            "robot0:slide0": 0.403,
            "robot0:slide1": 0.481,
            "robot0:slide2": 0.0,
        }

        super().__init__(
            model_path=xml_path,
            n_substeps=20,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            render_mode=render_mode,
            gripper_extra_height=gripper_extra_height,
            block_gripper=block_gripper,
            has_object=False,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            **kwargs,
        )

    def check_distance(self, achieved_goal, goal):
        """Compute Euclidean distance between achieved and desired goals."""
        return np.linalg.norm(achieved_goal - goal, axis=-1)

    def check_horizontal_distance(self, achieved_goal, goal):
        """Compute XY-plane distance between achieved and desired goals."""
        return np.linalg.norm(achieved_goal[..., :2] - goal[..., :2], axis=-1)

    def get_hammer_tip_position(self):
        """Return world position of the hammer tip site."""
        return self.data.site("hammer_tip").xpos.copy()

    def get_hammer_tip_velocity(self):
        """Return hammer tip linear velocity in world coordinates."""
        site_id = self.model.site("hammer_tip").id
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

    def get_gripper_position(self):
        """Return world position of the gripper site."""
        return self.data.site("robot0:grip").xpos.copy()

    def get_gripper_state(self):
        """Return gripper finger joint positions and velocities."""
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model,
            self.data,
            self._model_names.joint_names,
        )
        return robot_qpos[-2:].copy(), robot_qvel[-2:].copy()

    def get_hammer_handle_position(self):
        """Return world position of the hammer handle site."""
        return self.data.site("hammer_handle_site").xpos.copy()

    def _get_obs(self):
        """Return observation with hammer tip as achieved goal."""
        obs = super()._get_obs()
        obs["achieved_goal"] = self.get_hammer_tip_position()
        return obs

    def _move_goal_marker(self):
        """Move visible target marker to current goal position."""
        target_site_id = self.model.site("target0").id
        target_body_id = int(self.model.site("target0").bodyid[0])
        target_body_pos = self.data.body(target_body_id).xpos
        self.model.site_pos[target_site_id] = self.goal - target_body_pos
        mujoco.mj_forward(self.model, self.data)

    def _render_callback(self):
        """Update render-time marker positions before drawing."""
        self._move_goal_marker()

    def reset(self, *args, **kwargs):
        """Reset environment and synchronize goal-related observation fields.

        Args:
            *args: Positional reset arguments.
            **kwargs: Keyword reset arguments.

        Returns:
            Tuple of observation and info dict.
        """
        obs, info = super().reset(*args, **kwargs)
        self._move_goal_marker()
        obs["achieved_goal"] = self.get_hammer_tip_position()
        obs["desired_goal"] = self.goal.copy()
        return obs, info

    def _sample_goal(self):
        """Sample a discrete mole-hole goal location."""
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
        """Check whether current hammer-tip motion counts as a valid strike.

        Args:
            achieved_goal: Current hammer tip position.
            goal: Current mole target position.

        Returns:
            Tuple of ``(is_hit, distance, horizontal_distance, height_error,
            speed, downward_alignment)``.
        """
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

    def get_hammer_orientation_quat(self) -> np.ndarray:
        """Return canonical hammer orientation used by all task variants."""
        return self.hammer_quat.copy()
