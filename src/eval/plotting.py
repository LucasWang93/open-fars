"""Generate comparison plots from experiment results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def plot_comparison(
    baseline_scores: list[float],
    treatment_scores: list[float],
    metric_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    groups = ["Baseline", "Treatment"]
    means = [
        sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
        sum(treatment_scores) / len(treatment_scores) if treatment_scores else 0,
    ]

    import statistics
    stds = [
        statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0,
        statistics.stdev(treatment_scores) if len(treatment_scores) > 1 else 0,
    ]

    bars = ax.bar(groups, means, yerr=stds, capsize=8, color=["#4878CF", "#E36B5B"], alpha=0.85)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}: Baseline vs Treatment")
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 1)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    LOGGER.info("saved plot to %s", output_path)
