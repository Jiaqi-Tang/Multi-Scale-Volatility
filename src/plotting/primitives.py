"""Reusable matplotlib plotting primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.globals.columns import COMPONENT, COMPONENT_TYPE, INDEX, ORIGINAL, SERIES
from src.globals.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_ORDER, SERIES_SHUFFLE
from src.plotting.stats import (
    autocorrelation,
    compressed_layer_autocorrelation,
    ecdf,
    normal_quantiles_for_values,
)
from src.plotting.style import (
    FIGURE_DPI,
    FINAL_COLOR,
    FINAL_DARK_COLOR,
    GAUSSIAN_COLOR,
    GAUSSIAN_DARK_COLOR,
    GRID_FIGURE_DPI,
    SERIES_COLORS,
    SERIES_LABELS,
    SHUFFLE_COLOR,
)
from src.scale_utils import compress_component, decomposition_components


def plot_return_line(returns: np.ndarray, output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(np.arange(len(returns)), returns, linewidth=0.25, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Log return")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


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
    add_mean_median_lines(ax, gaussian_returns, "Gaussian", GAUSSIAN_DARK_COLOR)

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
    ax.plot(final_x, final_y, linewidth=1.2, label="EUR/USD final", color=FINAL_COLOR)
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


def plot_acf_comparison(
    final_values: np.ndarray,
    shuffle_values: np.ndarray,
    gaussian_values: np.ndarray,
    output_path: Path,
    max_lag: int,
    title: str,
    transform_label: str,
) -> Path:
    final_acf = autocorrelation(final_values, max_lag)
    shuffle_acf = autocorrelation(shuffle_values, max_lag)
    gaussian_acf = autocorrelation(gaussian_values, max_lag)
    lags = np.arange(1, max_lag + 1)
    band = 1.96 / np.sqrt(len(final_values))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lags, final_acf, linewidth=1.2, label="EUR/USD final", color=FINAL_COLOR)
    ax.plot(
        lags,
        shuffle_acf,
        linewidth=1.0,
        label="Shuffled baseline",
        color=SHUFFLE_COLOR,
        alpha=0.9,
    )
    ax.plot(
        lags,
        gaussian_acf,
        linewidth=1.0,
        label="Gaussian baseline",
        color=GAUSSIAN_COLOR,
        alpha=0.9,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax.axhline(band, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-band, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel(f"ACF of {transform_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


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
    gap_columns = ["entropy_gap_shuffle", "entropy_gap_gaussian"]
    if ordered[gap_columns].isna().any().any():
        missing = ordered[ordered[gap_columns].isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing entropy gap values for components: {missing}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        components,
        ordered["entropy_gap_shuffle"].astype(float).to_numpy(),
        marker="o",
        linewidth=1.7,
        markersize=4.5,
        label="shuffle - final",
        color=SHUFFLE_COLOR,
    )
    ax.plot(
        components,
        ordered["entropy_gap_gaussian"].astype(float).to_numpy(),
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

        axis.bar(
            patterns,
            percentages,
            color=SERIES_COLORS[series],
            alpha=0.8,
            width=0.72,
        )
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
    fig.suptitle(
        f"Ordinal Pattern Distribution: {SERIES_LABELS[series]}",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
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


def plot_decomposition_layers(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    k: int,
) -> Path:
    layers = decomposition_components(k, include_original=True)
    x = frame[INDEX].to_numpy()

    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(16, 24),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16)

    for axis, layer in zip(axes, layers):
        axis.plot(x, frame[layer].to_numpy(), linewidth=0.2, alpha=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.55)
        axis.set_ylabel(layer)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[-1].set_xlabel("Observation index")
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_abs_component_correlation_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    k: int,
) -> Path:
    components = decomposition_components(k, include_original=False)
    abs_components = frame[components].abs()
    correlation = abs_components.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(
        correlation.to_numpy(),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticklabels(components)

    for row_index, component_row in enumerate(components):
        for column_index, component_column in enumerate(components):
            value = float(correlation.loc[component_row, component_column])
            text_color = "white" if value < 0.55 else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

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
    final_correlation = final_frame[components].abs().corr(method="pearson")
    baseline_correlation = baseline_frame[components].abs().corr(method="pearson")
    difference = final_correlation - baseline_correlation

    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(
        difference.to_numpy(),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticklabels(components)

    for row_index, component_row in enumerate(components):
        for column_index, component_column in enumerate(components):
            value = float(difference.loc[component_row, component_column])
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Final minus shuffled correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
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


def plot_layer_acf_grid(
    final_frame: pd.DataFrame,
    shuffle_frame: pd.DataFrame,
    gaussian_frame: pd.DataFrame,
    output_path: Path,
    layers: list[str],
    max_lag: int,
    absolute: bool,
    title: str,
) -> Path:
    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(16, 22),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(f"{title} (compressed repeated block values for plotting)", fontsize=16)

    for axis, layer in zip(axes, layers):
        final_values = final_frame[layer].to_numpy()
        shuffle_values = shuffle_frame[layer].to_numpy()
        gaussian_values = gaussian_frame[layer].to_numpy()
        compressed_n = len(compress_component(final_values, layer))
        band = 1.96 / np.sqrt(compressed_n)
        if absolute:
            final_values = np.abs(final_values)
            shuffle_values = np.abs(shuffle_values)
            gaussian_values = np.abs(gaussian_values)

        final_lags, final_acf = compressed_layer_autocorrelation(
            final_values, layer, max_lag
        )
        shuffle_lags, shuffle_acf = compressed_layer_autocorrelation(
            shuffle_values, layer, max_lag
        )
        gaussian_lags, gaussian_acf = compressed_layer_autocorrelation(
            gaussian_values, layer, max_lag
        )

        axis.plot(final_lags, final_acf, linewidth=1.0, label="EUR/USD")
        axis.plot(
            shuffle_lags,
            shuffle_acf,
            linewidth=0.9,
            label="Shuffled",
            alpha=0.9,
        )
        axis.plot(
            gaussian_lags,
            gaussian_acf,
            linewidth=0.9,
            label="Gaussian",
            alpha=0.9,
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.75)
        axis.axhline(band, color="black", linestyle="--", linewidth=0.7, alpha=0.45)
        axis.axhline(-band, color="black", linestyle="--", linewidth=0.7, alpha=0.45)
        axis.set_ylabel(layer)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Lag")
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
    plt.close(fig)
    return output_path


def metric_components(k: int, include_approximation: bool) -> list[str]:
    components = [f"D_{scale:02d}" for scale in range(1, k + 1)]
    if include_approximation:
        components.append(f"A_{k:02d}")
    return components


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
    ax.plot([], [], color=color, linestyle="--", label=f"{label_prefix} median")
