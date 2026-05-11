from __future__ import annotations

from whack_a_mole.envs.end_to_end import EndToEndEnv
from whack_a_mole.envs.hammer_use import HammerUseEnv
from whack_a_mole.envs.pickup import PickupEnv
from whack_a_mole.envs.wrappers import HammerUseObservationWrapper, PickupObservationWrapper


def create_env(render_mode=None, reward_type: str = "dense", task: str = "hammer_use"):
    """Create a wrapped environment for a specific task.

    Args:
        render_mode: Gymnasium render mode, such as ``None`` or ``rgb_array``.
        reward_type: Reward type. Only ``dense`` is supported.
        task: Task name. One of ``pickup``, ``hammer_use``, or ``end_to_end``.

    Returns:
        A task-specific wrapped Gymnasium environment.

    Raises:
        ValueError: If reward type or task is unsupported.
    """
    if reward_type != "dense":
        raise ValueError("Only dense reward is supported for this setup")

    if task == "pickup":
        return PickupObservationWrapper(PickupEnv(render_mode=render_mode, reward_type=reward_type))
    if task == "hammer_use":
        return HammerUseObservationWrapper(HammerUseEnv(render_mode=render_mode, reward_type=reward_type))
    if task == "end_to_end":
        return PickupObservationWrapper(EndToEndEnv(render_mode=render_mode, reward_type=reward_type))
    raise ValueError(f"Unsupported task '{task}'")
