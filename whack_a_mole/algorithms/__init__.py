from whack_a_mole.algorithms.base import EvalResult, RLAlgorithm, TrainConfig, TrainResult
from whack_a_mole.algorithms.ppo_actor import PPOActor
from whack_a_mole.algorithms.q_learning import QLearningActor, QNet
from whack_a_mole.algorithms.reinforce import Reinforce, StochasticPolicyNet


def make_algorithm(name: str, **kwargs) -> RLAlgorithm:
    normalized = name.strip().lower()
    if normalized == "ppo":
        return PPOActor(**kwargs)
    if normalized == "reinforce":
        return Reinforce(**kwargs)
    if normalized in {"q_learning", "qlearning", "q-learning"}:
        return QLearningActor(**kwargs)
    raise ValueError(f"Unsupported algorithm '{name}'")


__all__ = [
    "RLAlgorithm",
    "TrainConfig",
    "TrainResult",
    "EvalResult",
    "PPOActor",
    "Reinforce",
    "StochasticPolicyNet",
    "QNet",
    "QLearningActor",
    "make_algorithm",
]
