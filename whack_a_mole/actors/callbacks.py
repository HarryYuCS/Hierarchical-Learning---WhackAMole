from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class SB3MetricsCallback(BaseCallback):
    """Collect selected SB3 logger metrics during training."""

    def __init__(self, metric_keys: list[str], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.metric_keys = metric_keys
        self.metrics: dict[str, list[float]] = {k: [] for k in metric_keys}
        self.timesteps: list[int] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logger_values = getattr(self.model.logger, "name_to_value", {})
        self.timesteps.append(int(self.num_timesteps))
        for key in self.metric_keys:
            value = logger_values.get(key, None)
            if value is None:
                self.metrics[key].append(float("nan"))
            else:
                self.metrics[key].append(float(value))
