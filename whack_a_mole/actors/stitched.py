from __future__ import annotations

import numpy as np

from whack_a_mole.actors.base import Actor
from whack_a_mole.envs.wrappers import hammer_use_obs_from_full_obs


class StitchedPPOActor(Actor):
    """Actor that switches between pickup and hammer-use PPO policies.

    Attributes:
        pickup_actor: Policy used before the hammer is grasped.
        hammer_use_actor: Policy used after grasp/lift is detected.
        lift_threshold: Distance threshold used by the switch heuristic.
    """

    def __init__(self, pickup_actor, hammer_use_actor, lift_threshold: float = 0.5):
        """Initialize stitched actor.

        Args:
            pickup_actor: First-stage actor for pickup behavior.
            hammer_use_actor: Second-stage actor for hammer-use behavior.
            lift_threshold: Gripper-to-handle distance threshold.
        """
        self.pickup_actor = pickup_actor
        self.hammer_use_actor = hammer_use_actor
        self.lift_threshold = lift_threshold
        self.active_policy = "pickup"
        self.switch_count = 0

    def _is_hammer_grasped(self, obs) -> bool:
        """Infer whether the hammer is grasped/lifted from observation features.

        Args:
            obs: Observation dict from pickup/end-to-end wrapper.

        Returns:
            True when switch condition is met, otherwise False.
        """
        if not isinstance(obs, dict):
            return False
        features = np.asarray(obs.get("observation", []), dtype=np.float32)
        if features.shape[0] < 25:
            return False
        held_flag = float(features[-1])
        return bool(held_flag > 0.5)

    def predict(self, obs, deterministic: bool = True):
        """Predict an action from the active sub-policy.

        Args:
            obs: Environment observation.
            deterministic: Whether to use deterministic action selection.

        Returns:
            Action predicted by either pickup or hammer-use policy.
        """
        if self._is_hammer_grasped(obs):
            if self.active_policy != "hammer_use":
                self.switch_count += 1
            self.active_policy = "hammer_use"
            return self.hammer_use_actor.predict(hammer_use_obs_from_full_obs(obs), deterministic=deterministic)
        self.active_policy = "pickup"
        return self.pickup_actor.predict(obs, deterministic=deterministic)
