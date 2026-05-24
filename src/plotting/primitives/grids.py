"""Grid plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.style import (
    FINAL_COLOR,
    FINAL_DARK_COLOR,
    GAUSSIAN_COLOR,
    GRID_FIGURE_DPI,
    SERIES_COLORS,
    SERIES_LABELS,
)
from src.scale_utils import decomposition_components
from src.stats import normal_quantiles_for_values


def plot_entropy_pattern_distribution_grid(
    pattern_counts: dict[str, dict[str, int]],
    output_path: Path,
    series: str,
    k: int,
) -> Path:
    components = decomposition_components(k, include_original=False)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey=True)
    axes_flat = axes.ravel()
    global_max = 0.0
    component_percentages: dict[str, tuple[list[str], np.ndarray]] = {}

    for component in components:
        counts = pattern_counts[component]
        patterns = sorted(counts)
        total = sum(counts.values())
        if total <= 0:
            raise ValueError(f"Pattern counts sum to zero for {series} {component}")
        percentages = np.array([counts[pattern] / total for pattern in patterns])
        component_percentages[component] = (patterns, percentages)
        global_max = max(global_max, float(percentages.max()))

    for axis, component in zip(axes_flat, components):
        patterns, percentages = component_percentages[component]
        axis.bar(patterns, percentages, color=SERIES_COLORS[series], alpha=0.8, width=0.72)
        axis.axhline(
            1.0 / len(patterns),
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
        )
        axis.set_title(component)
        axis.set_ylim(0.0, max(0.25, global_max * 1.18))
        axis.tick_params(axis="x", labelrotation=35)

    for axis in axes_flat[len(components):]:
        axis.axis("off")

    for axis in axes[:, 0]:
        axis.set_ylabel("Pattern share")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="Uniform share",
        )
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"Ordinal Pattern Distribution: {SERIES_LABELS[series]}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_layer_histogram_grid(
    final_frame: pd.DataFrame,
    gaussian_frame: pd.DataFrame,
    output_path: Path,
    k: int,
) -> Path:
    layers = decomposition_components(k, include_original=False)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes_flat = axes.ravel()

    for axis, layer in zip(axes_flat, layers):
        final_values = final_frame[layer].to_numpy()
        gaussian_values = gaussian_frame[layer].to_numpy()
        axis.hist(
            final_values,
            bins=180,
            density=True,
            alpha=0.55,
            label="EUR/USD",
            color=FINAL_COLOR,
        )
        axis.hist(
            gaussian_values,
            bins=180,
            density=True,
            alpha=0.45,
            label="Gaussian",
            color=GAUSSIAN_COLOR,
        )
        axis.axvline(np.mean(final_values), color=FINAL_DARK_COLOR, linewidth=0.8)
        axis.axvline(
            np.median(final_values),
            color=FINAL_DARK_COLOR,
            linestyle="--",
            linewidth=0.8,
        )
        axis.set_title(layer)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

    for axis in axes_flat[len(layers):]:
        axis.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Layer Distributions: EUR/USD vs Gaussian Baseline", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_layer_qq_grid(
    final_frame: pd.DataFrame,
    output_path: Path,
    k: int,
    quantile_points: int = 2_000,
) -> Path:
    layers = decomposition_components(k, include_original=False)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes_flat = axes.ravel()

    for axis, layer in zip(axes_flat, layers):
        gaussian_quantiles, empirical_quantiles = normal_quantiles_for_values(
            final_frame[layer].to_numpy(),
            quantile_points,
        )
        axis.scatter(
            gaussian_quantiles,
            empirical_quantiles,
            s=3,
            alpha=0.3,
            color=FINAL_COLOR,
            edgecolors="none",
        )
        low = min(float(gaussian_quantiles.min()), float(empirical_quantiles.min()))
        high = max(float(gaussian_quantiles.max()), float(empirical_quantiles.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=0.8, alpha=0.75)
        axis.set_title(layer)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))

    for axis in axes_flat[len(layers):]:
        axis.axis("off")

    fig.suptitle("Layer QQ Plots vs N(0, Layer Variance)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
    plt.close(fig)
    return output_path

