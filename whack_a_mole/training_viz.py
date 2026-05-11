from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from whack_a_mole.actors.base import TrainResult


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
    timesteps = train_result.metadata.get("timesteps", list(range(len(next(iter(metrics.values()), [])))))

    valid_items = [(k, v) for k, v in metrics.items() if isinstance(v, list) and len(v) > 0]
    if not valid_items:
        raise ValueError("No metric history found in train_result.metadata['metrics']")

    fig, axes = plt.subplots(len(valid_items), 1, figsize=(10, 3 * len(valid_items)), squeeze=False)
    axes = axes.ravel()
    for ax, (name, values) in zip(axes, valid_items):
        ax.plot(timesteps[: len(values)], values)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("timesteps")
    fig.suptitle(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
