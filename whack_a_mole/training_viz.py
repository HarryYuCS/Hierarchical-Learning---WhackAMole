from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from whack_a_mole.actors.base import TrainResult


PREFERRED_METRICS = [
    "rollout/ep_rew_mean",
    "train/loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/critic_loss",
    "train/actor_loss",
    "train/ent_coef",
]


def _moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    if values.size < window or window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_training_metrics(train_result: TrainResult, output_path: str | Path, title: str = "Training Metrics") -> Path:
    """Plot tracked training metrics from TrainResult metadata.

    Args:
        train_result: Result returned by actor.train(...).
        output_path: Path to save the figure image.
        title: Plot title.

    Returns:
        Path to the saved plot image.
    """
    metrics = train_result.metadata.get("metrics", {})

    def unpack_series(series):
        if isinstance(series, dict) and "values" in series:
            vals = series.get("values", [])
            ts = series.get("timesteps", list(range(len(vals))))
            return ts, vals
        if isinstance(series, list):
            return list(range(len(series))), series
        return [], []

    available = {}
    for name, series in metrics.items():
        ts, vals = unpack_series(series)
        if len(vals) > 0:
            available[name] = {"timesteps": ts, "values": vals}

    if train_result.episode_rewards:
        available.setdefault(
            "episode_rewards/mean",
            {
                "timesteps": list(range(len(train_result.episode_rewards))),
                "values": list(train_result.episode_rewards),
            },
        )

    if train_result.losses:
        available.setdefault(
            "train_result/losses",
            {
                "timesteps": list(range(len(train_result.losses))),
                "values": list(train_result.losses),
            },
        )

    preferred = [(k, available[k]) for k in PREFERRED_METRICS if k in available]
    selected = preferred if preferred else list(available.items())

    if not selected and train_result.losses:
        selected = [("train/loss", {"timesteps": list(range(len(train_result.losses))), "values": train_result.losses})]

    if not selected:
        raise ValueError("No metric history found in train_result metadata")

    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 3 * len(selected)), squeeze=False)
    axes = axes.ravel()
    for ax, (name, series) in zip(axes, selected):
        ts = np.asarray(series["timesteps"], dtype=np.float64)
        values_arr = np.asarray(series["values"], dtype=np.float64)
        n = min(ts.shape[0], values_arr.shape[0])
        x = ts[:n]
        values_arr = values_arr[:n]

        finite_mask = np.isfinite(values_arr)
        x_f = x[finite_mask]
        y_f = values_arr[finite_mask]
        if y_f.size == 0:
            continue

        ax.plot(x_f, y_f, alpha=0.35, label="raw")
        y_s = _moving_average(y_f, window=10)
        if y_s.size < y_f.size:
            x_s = x_f[-y_s.size :]
            ax.plot(x_s, y_s, linewidth=2.0, label="ma(10)")

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("timesteps")
    fig.suptitle(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
