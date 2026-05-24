"""QQ-plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.style import FIGURE_DPI, FINAL_COLOR
from src.stats import normal_quantiles_for_values


def plot_qq_against_zero_mean_gaussian(
    returns: np.ndarray,
    output_path: Path,
    quantile_points: int = 2_000,
) -> Path:
    gaussian_quantiles, empirical_quantiles = normal_quantiles_for_values(
        returns,
        quantile_points,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        gaussian_quantiles,
        empirical_quantiles,
        s=4,
        alpha=0.35,
        color=FINAL_COLOR,
        edgecolors="none",
    )
    low = min(float(gaussian_quantiles.min()), float(empirical_quantiles.min()))
    high = max(float(gaussian_quantiles.max()), float(empirical_quantiles.max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("QQ Plot: EUR/USD Final Returns vs N(0, empirical variance)")
    ax.set_xlabel("Gaussian theoretical quantile")
    ax.set_ylabel("EUR/USD empirical quantile")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path

