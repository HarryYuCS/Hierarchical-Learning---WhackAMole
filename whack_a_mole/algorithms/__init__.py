from whack_a_mole.actors import Actor as RLAlgorithm
from whack_a_mole.actors import EvalResult, PPOActor, SACActor, TrainConfig, TrainResult, TrainableActor


def make_algorithm(name: str, **kwargs) -> RLAlgorithm:
    normalized = name.strip().lower()
    if normalized == "ppo":
        return PPOActor(**kwargs)
    if normalized == "sac":
        return SACActor(**kwargs)
    raise ValueError(f"Unsupported algorithm '{name}'")


__all__ = [
    "RLAlgorithm",
    "TrainableActor",
    "TrainConfig",
    "TrainResult",
    "EvalResult",
    "PPOActor",
    "SACActor",
    "make_algorithm",
]
