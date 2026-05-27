"""Distribution plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multi_scale_volatility.plotting.style import (
    FIGURE_DPI,
    FINAL_COLOR,
    FINAL_DARK_COLOR,
    GAUSSIAN_COLOR,
    GAUSSIAN_DARK_COLOR,
)
from multi_scale_volatility.stats import ecdf


def plot_histogram_comparison(
    final_returns: np.ndarray,
    gaussian_returns: np.ndarray,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        final_returns,
        bins=300,
        density=True,
        alpha=0.55,
        label="EUR/USD final",
        color=FINAL_COLOR,
    )
    ax.hist(
        gaussian_returns,
        bins=300,
        density=True,
        alpha=0.45,
        label="Gaussian baseline",
        color=GAUSSIAN_COLOR,
    )

    add_mean_median_lines(ax, final_returns, "EUR/USD", FINAL_DARK_COLOR)
    add_mean_median_lines(ax, gaussian_returns,
                          "Gaussian", GAUSSIAN_DARK_COLOR)

    ax.set_title("Distribution of 5m Log Returns")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_ecdf_comparison(
    final_returns: np.ndarray,
    gaussian_returns: np.ndarray,
    output_path: Path,
) -> Path:
    final_x, final_y = ecdf(final_returns)
    gaussian_x, gaussian_y = ecdf(gaussian_returns)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(final_x, final_y, linewidth=1.2,
            label="EUR/USD final", color=FINAL_COLOR)
    ax.plot(
        gaussian_x,
        gaussian_y,
        linewidth=1.2,
        label="Gaussian baseline",
        color=GAUSSIAN_COLOR,
    )
    ax.set_title("Empirical CDF of 5m Log Returns")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Cumulative probability")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def add_mean_median_lines(
    ax: plt.Axes,
    returns: np.ndarray,
    label_prefix: str,
    color: str,
) -> None:
    mean = float(np.mean(returns))
    median = float(np.median(returns))
    ax.axvline(mean, color=color, linestyle="-", linewidth=1.0, alpha=0.85)
    ax.axvline(median, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
    ax.plot([], [], color=color, linestyle="-", label=f"{label_prefix} mean")
    ax.plot([], [], color=color, linestyle="--",
            label=f"{label_prefix} median")
