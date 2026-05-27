"""Heatmap plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.plotting.style import FIGURE_DPI
from multi_scale_volatility.scale_utils import decomposition_components
from multi_scale_volatility.stats import (
    absolute_component_correlation,
    absolute_component_correlation_difference,
)


def plot_abs_component_correlation_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    k: int,
) -> Path:
    components = decomposition_components(k, include_original=False)
    correlation = absolute_component_correlation(frame, components)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(
        correlation.to_numpy(),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="equal",
    )
    _label_heatmap_axes(ax, components)
    ax.set_title(title)
    _annotate_heatmap(ax, correlation, threshold=0.55, low_color="white")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation of absolute components")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_abs_component_correlation_difference_heatmap(
    final_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    k: int,
) -> Path:
    components = decomposition_components(k, include_original=False)
    difference = absolute_component_correlation_difference(
        final_frame,
        baseline_frame,
        components,
    )

    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(
        difference.to_numpy(),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        aspect="equal",
    )
    _label_heatmap_axes(ax, components)
    ax.set_title(title)
    _annotate_heatmap(ax, difference, threshold=0.5, high_abs_color="white")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Final minus shuffled correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def _label_heatmap_axes(ax: plt.Axes, components: list[str]) -> None:
    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticklabels(components)


def _annotate_heatmap(
    ax: plt.Axes,
    values: pd.DataFrame,
    threshold: float,
    low_color: str = "black",
    high_abs_color: str | None = None,
) -> None:
    for row_index, component_row in enumerate(values.index):
        for column_index, component_column in enumerate(values.columns):
            value = float(values.loc[component_row, component_column])
            if high_abs_color is not None:
                text_color = high_abs_color if abs(
                    value) > threshold else "black"
            else:
                text_color = low_color if value < threshold else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )
