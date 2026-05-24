"""Metric line-plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.columns import COMPONENT, SERIES
from src.config.metric_columns import ENTROPY_GAP_GAUSSIAN, ENTROPY_GAP_SHUFFLE
from src.config.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_ORDER, SERIES_SHUFFLE
from src.plotting.style import (
    FIGURE_DPI,
    GAUSSIAN_COLOR,
    SERIES_COLORS,
    SHUFFLE_COLOR,
)
from src.scale_utils import decomposition_components


def plot_series_metric(
    frame: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    components: list[str],
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))

    for series in SERIES_ORDER:
        series_frame = (
            frame[frame[SERIES] == series]
            .set_index(COMPONENT)
            .reindex(components)
        )
        if series_frame[metric].isna().any():
            missing = series_frame[series_frame[metric].isna()].index.tolist()
            raise ValueError(f"Missing {metric} values for {series}: {missing}")
        ax.plot(
            components,
            series_frame[metric].astype(float).to_numpy(),
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            label=series,
            color=SERIES_COLORS[series],
        )

    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_baseline_difference_metric(
    frame: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    components: list[str],
    final_minus_baseline: bool,
) -> Path:
    wide = frame.pivot(index=COMPONENT, columns=SERIES, values=metric).reindex(components)
    required_series = list(SERIES_ORDER)
    missing_series = [series for series in required_series if series not in wide.columns]
    if missing_series:
        raise ValueError(f"Missing series for {metric}: {missing_series}")
    if wide[required_series].isna().any().any():
        missing_components = wide[wide[required_series].isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing {metric} values for components: {missing_components}")

    if final_minus_baseline:
        shuffle_values = wide[SERIES_FINAL] - wide[SERIES_SHUFFLE]
        gaussian_values = wide[SERIES_FINAL] - wide[SERIES_GAUSSIAN]
        shuffle_label = "final - shuffle"
        gaussian_label = "final - gaussian"
    else:
        shuffle_values = wide[SERIES_SHUFFLE] - wide[SERIES_FINAL]
        gaussian_values = wide[SERIES_GAUSSIAN] - wide[SERIES_FINAL]
        shuffle_label = "shuffle - final"
        gaussian_label = "gaussian - final"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        components,
        shuffle_values.astype(float).to_numpy(),
        marker="o",
        linewidth=1.7,
        markersize=4.5,
        label=shuffle_label,
        color=SHUFFLE_COLOR,
    )
    ax.plot(
        components,
        gaussian_values.astype(float).to_numpy(),
        marker="o",
        linewidth=1.7,
        markersize=4.5,
        label=gaussian_label,
        color=GAUSSIAN_COLOR,
    )
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_entropy_metric(
    frame: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    k: int,
) -> Path:
    return plot_series_metric(
        frame,
        output_path,
        metric=metric,
        title=title,
        ylabel=ylabel,
        components=decomposition_components(k, include_original=False),
    )


def plot_entropy_gaps(frame: pd.DataFrame, output_path: Path, k: int) -> Path:
    components = decomposition_components(k, include_original=False)
    ordered = frame.set_index(COMPONENT).reindex(components)
    gap_columns = [ENTROPY_GAP_SHUFFLE, ENTROPY_GAP_GAUSSIAN]
    if ordered[gap_columns].isna().any().any():
        missing = ordered[ordered[gap_columns].isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing entropy gap values for components: {missing}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        components,
        ordered[ENTROPY_GAP_SHUFFLE].astype(float).to_numpy(),
        marker="o",
        linewidth=1.7,
        markersize=4.5,
        label="shuffle - final",
        color=SHUFFLE_COLOR,
    )
    ax.plot(
        components,
        ordered[ENTROPY_GAP_GAUSSIAN].astype(float).to_numpy(),
        marker="o",
        linewidth=1.7,
        markersize=4.5,
        label="gaussian - final",
        color=GAUSSIAN_COLOR,
    )
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.8)
    ax.set_title("Normalized Entropy Gaps from Baselines")
    ax.set_xlabel("Component")
    ax.set_ylabel("Baseline minus final normalized entropy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_volatility_metric(
    frame: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    k: int,
    include_approximation: bool,
) -> Path:
    return plot_series_metric(
        frame,
        output_path,
        metric=metric,
        title=title,
        ylabel=ylabel,
        components=metric_components(k, include_approximation),
    )


def plot_volatility_difference_metric(
    frame: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    k: int,
    include_approximation: bool,
) -> Path:
    return plot_baseline_difference_metric(
        frame,
        output_path,
        metric=metric,
        title=title,
        ylabel=ylabel,
        components=metric_components(k, include_approximation),
        final_minus_baseline=True,
    )


def metric_components(k: int, include_approximation: bool) -> list[str]:
    components = [f"D_{scale:02d}" for scale in range(1, k + 1)]
    if include_approximation:
        components.append(f"A_{k:02d}")
    return components
