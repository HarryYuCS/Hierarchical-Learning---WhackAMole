from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class SB3MetricsCallback(BaseCallback):
    """Collect selected SB3 logger metrics during training."""

    def __init__(self, metric_keys: list[str], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.metric_keys = metric_keys
        self.metrics: dict[str, dict[str, list[float]]] = {
            k: {"timesteps": [], "values": []} for k in metric_keys
        }
        self.last_seen: dict[str, int] = {k: -1 for k in metric_keys}

    def _on_step(self) -> bool:
        logger_values = getattr(self.model.logger, "name_to_value", {})
        current_t = int(self.num_timesteps)
        for key in self.metric_keys:
            value = logger_values.get(key, None)
            if value is None:
                continue
            if self.last_seen[key] == current_t:
                continue
            self.metrics[key]["timesteps"].append(current_t)
            self.metrics[key]["values"].append(float(value))
            self.last_seen[key] = current_t
        return True
