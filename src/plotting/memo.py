"""Polished figures for Memo.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from src.globals.constants import DEFAULT_K
from src.globals.paths import (
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    GAUSSIAN_RETURNS_CSV,
    LAYER_ENTROPY_CSV,
    MEMO_PLOTS_DIR,
    SHUFFLE_RETURNS_CSV,
    SHUFFLE_DECOMPOSITION_CSV,
    VOLATILITY_CSV,
)
from src.globals.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_SHUFFLE
from src.plotting.primitives.distributions import add_mean_median_lines
from src.plotting.readers import read_decomposition, read_returns
from src.plotting.style import (
    FIGURE_DPI,
    FINAL_COLOR,
    FINAL_DARK_COLOR,
    GAUSSIAN_COLOR,
    GAUSSIAN_DARK_COLOR,
    SHUFFLE_COLOR,
)
from src.stats import (
    absolute_component_correlation,
    absolute_component_correlation_difference,
    autocorrelation,
    normal_quantiles_for_values,
)
from src.utils.validation import require_positive_k
from src.scale_utils import decomposition_components


@dataclass(frozen=True)
class MemoPlotPaths:
    final_returns_csv: Path = FINAL_RETURNS_CSV
    shuffle_returns_csv: Path = SHUFFLE_RETURNS_CSV
    gaussian_returns_csv: Path = GAUSSIAN_RETURNS_CSV
    final_decomposition_csv: Path = FINAL_DECOMPOSITION_CSV
    shuffle_decomposition_csv: Path = SHUFFLE_DECOMPOSITION_CSV
    volatility_csv: Path = VOLATILITY_CSV
    layer_entropy_csv: Path = LAYER_ENTROPY_CSV
    output_dir: Path = MEMO_PLOTS_DIR


def create_memo_plots(
    paths: MemoPlotPaths | None = None,
    k: int = DEFAULT_K,
) -> list[Path]:
    paths = paths or MemoPlotPaths()
    require_positive_k(k)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    final_returns = read_returns(paths.final_returns_csv)
    shuffle_returns = read_returns(paths.shuffle_returns_csv)
    gaussian_returns = read_returns(paths.gaussian_returns_csv)
    final_decomposition = read_decomposition(paths.final_decomposition_csv, k)
    shuffle_decomposition = read_decomposition(paths.shuffle_decomposition_csv, k)
    volatility = pd.read_csv(paths.volatility_csv)
    entropy = pd.read_csv(paths.layer_entropy_csv)

    return [
        plot_memo_decomposition_example(
            final_decomposition,
            paths.output_dir / "figure_01_decomposition_example.png",
        ),
        plot_memo_return_distribution(
            final_returns,
            gaussian_returns,
            paths.output_dir / "figure_02_return_distribution.png",
        ),
        plot_memo_abs_return_acf(
            final_returns,
            shuffle_returns,
            gaussian_returns,
            paths.output_dir / "figure_03_abs_return_acf.png",
        ),
        plot_memo_energy_profile(
            volatility,
            paths.output_dir / "figure_04_energy_profile.png",
        ),
        plot_memo_cross_scale_correlation(
            final_decomposition,
            shuffle_decomposition,
            paths.output_dir / "figure_05_cross_scale_correlation.png",
            k=k,
        ),
        plot_memo_cross_scale_correlation(
            final_decomposition,
            shuffle_decomposition,
            paths.output_dir / "figure_05_cross_scale_correlation_same_scale.png",
            k=k,
            same_scale=True,
        ),
        plot_memo_entropy_profile(
            entropy,
            paths.output_dir / "figure_06_entropy_profile.png",
            k=k,
        )
    ]


def plot_memo_decomposition_example(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    layers = [
        ("original", "Original 5m return"),
        ("D_01", "Detail D_01"),
        ("D_03", "Detail D_03"),
        ("D_06", "Detail D_06"),
        ("D_09", "Detail D_09"),
        ("D_11", "Detail D_11"),
        ("A_11", "Approximation A_11"),
    ]
    x = frame["index"].to_numpy()

    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(14, 12),
        sharex=True,
        constrained_layout=True,
    )

    for axis, (column, label) in zip(axes, layers):
        values = frame[column].astype(float).to_numpy()
        axis.plot(
            x,
            values,
            color=FINAL_COLOR,
            linewidth=0.35,
            alpha=0.8,
            rasterized=True,
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.65)
        axis.set_ylabel(label)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[-1].set_xlabel("Observation index")
    fig.suptitle("EUR/USD Return Decomposition Across Representative Scales")
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_memo_return_distribution(
    final_returns: np.ndarray,
    gaussian_returns: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    histogram_axis, qq_axis = axes
    bins = 220
    histogram_range = (-0.0015, 0.0015)
    histogram_axis.hist(
        final_returns,
        bins=bins,
        range=histogram_range,
        density=True,
        alpha=0.58,
        color=FINAL_COLOR,
        label="EUR/USD",
    )
    histogram_axis.hist(
        gaussian_returns,
        bins=bins,
        range=histogram_range,
        density=True,
        alpha=0.45,
        color=GAUSSIAN_COLOR,
        label="Gaussian baseline",
    )
    add_mean_median_lines(histogram_axis, final_returns, "EUR/USD", FINAL_DARK_COLOR)
    add_mean_median_lines(
        histogram_axis,
        gaussian_returns,
        "Gaussian",
        GAUSSIAN_DARK_COLOR,
    )
    histogram_axis.set_title("Return Distribution (zoomed to [-0.0015, 0.0015])")
    histogram_axis.set_xlabel("5m log return")
    histogram_axis.set_ylabel("Density")
    histogram_axis.set_xlim(*histogram_range)
    histogram_axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    histogram_axis.legend()

    gaussian_quantiles, empirical_quantiles = normal_quantiles_for_values(
        final_returns,
        quantile_points=2_500,
    )
    qq_axis.scatter(
        gaussian_quantiles,
        empirical_quantiles,
        s=5,
        alpha=0.35,
        color=FINAL_COLOR,
        edgecolors="none",
    )
    low = min(float(gaussian_quantiles.min()), float(empirical_quantiles.min()))
    high = max(float(gaussian_quantiles.max()), float(empirical_quantiles.max()))
    qq_axis.plot([low, high], [low, high], color="black", linewidth=1.0, alpha=0.8)
    qq_axis.set_title("QQ Plot vs Variance-Matched Gaussian")
    qq_axis.set_xlabel("Gaussian theoretical quantile")
    qq_axis.set_ylabel("EUR/USD empirical quantile")
    qq_axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))

    fig.suptitle("EUR/USD Returns vs Gaussian Baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_memo_abs_return_acf(
    final_returns: np.ndarray,
    shuffle_returns: np.ndarray,
    gaussian_returns: np.ndarray,
    output_path: Path,
    max_lag: int = 1440,
) -> Path:
    final_acf = autocorrelation(np.abs(final_returns), max_lag)
    shuffle_acf = autocorrelation(np.abs(shuffle_returns), max_lag)
    gaussian_acf = autocorrelation(np.abs(gaussian_returns), max_lag)
    lags = np.arange(1, max_lag + 1)
    band = 1.96 / np.sqrt(len(final_returns))

    fig, axis = plt.subplots(figsize=(12, 5.8))
    axis.plot(lags, final_acf, color=FINAL_COLOR, linewidth=1.4, label="EUR/USD")
    axis.plot(
        lags,
        shuffle_acf,
        color=GAUSSIAN_DARK_COLOR,
        linewidth=1.0,
        alpha=0.75,
        label="Shuffled baseline",
    )
    axis.plot(
        lags,
        gaussian_acf,
        color=GAUSSIAN_COLOR,
        linewidth=1.0,
        alpha=0.75,
        label="Gaussian baseline",
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    axis.axhline(band, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axis.axhline(-band, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    for day in range(1, 6):
        axis.axvline(
            288 * day,
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.35,
        )

    axis.set_title("Autocorrelation of Absolute 5m Returns")
    axis.set_xlabel("Lag in 5m observations")
    axis.set_ylabel(r"$Corr(|r_i|, |r_{i-\ell}|)$")
    axis.set_xlim(1, max_lag)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_memo_energy_profile(
    volatility: pd.DataFrame,
    output_path: Path,
) -> Path:
    detail = volatility[volatility["component_type"] == "detail"].copy()
    components = [f"D_{index:02d}" for index in range(1, 12)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=False)
    share_axis, difference_axis = axes

    for series, color, label in [
        (SERIES_FINAL, FINAL_COLOR, "EUR/USD"),
        (SERIES_SHUFFLE, SHUFFLE_COLOR, "Shuffled baseline"),
        (SERIES_GAUSSIAN, GAUSSIAN_COLOR, "Gaussian baseline"),
    ]:
        series_frame = (
            detail[detail["series"] == series]
            .set_index("component")
            .reindex(components)
        )
        share_axis.plot(
            components,
            series_frame["detail_energy_share"].astype(float).to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=color,
            label=label,
        )

    wide = detail.pivot(
        index="component",
        columns="series",
        values="detail_energy_share",
    ).reindex(components)
    difference_axis.plot(
        components,
        (wide[SERIES_FINAL] - wide[SERIES_SHUFFLE]).astype(float).to_numpy(),
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        color=SHUFFLE_COLOR,
        label="EUR/USD - shuffled",
    )
    difference_axis.plot(
        components,
        (wide[SERIES_FINAL] - wide[SERIES_GAUSSIAN]).astype(float).to_numpy(),
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        color=GAUSSIAN_COLOR,
        label="EUR/USD - Gaussian",
    )
    difference_axis.axhline(0.0, color="black", linewidth=0.9, alpha=0.8)

    share_axis.set_title("Detail Energy Share")
    share_axis.set_ylabel("Share of detail-layer energy")
    share_axis.legend()
    share_axis.grid(axis="y", alpha=0.25)

    difference_axis.set_title("Excess Detail Energy Share")
    difference_axis.set_ylabel("EUR/USD minus baseline")
    difference_axis.legend()
    difference_axis.grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.set_xlabel("Component")
        axis.tick_params(axis="x", rotation=35)

    fig.suptitle("Volatility Energy Redistribution Across Scales")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_memo_cross_scale_correlation(
    final_decomposition: pd.DataFrame,
    shuffle_decomposition: pd.DataFrame,
    output_path: Path,
    k: int,
    same_scale: bool = False,
) -> Path:
    components = decomposition_components(k, include_original=False)
    final_corr = absolute_component_correlation(final_decomposition, components)
    excess_corr = absolute_component_correlation_difference(
        final_decomposition,
        shuffle_decomposition,
        components,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    final_axis, excess_axis = axes

    if same_scale:
        final_vmin, final_vmax = -1.0, 1.0
        excess_vmin, excess_vmax = -1.0, 1.0
        final_cmap = "Spectral_r"
        excess_cmap = "Spectral_r"
        title_suffix = " (shared [-1, 1] scale)"
    else:
        final_vmin, final_vmax = 0.0, 1.0
        max_abs_excess = float(np.nanmax(np.abs(excess_corr.to_numpy())))
        excess_limit = max(0.05, max_abs_excess)
        excess_vmin, excess_vmax = -excess_limit, excess_limit
        final_cmap = "viridis"
        excess_cmap = "coolwarm"
        title_suffix = ""

    final_image = final_axis.imshow(
        final_corr.to_numpy(),
        vmin=final_vmin,
        vmax=final_vmax,
        cmap=final_cmap,
        aspect="equal",
    )
    final_axis.set_title("EUR/USD Absolute Component Correlation")
    final_colorbar = fig.colorbar(final_image, ax=final_axis, fraction=0.046, pad=0.04)
    final_colorbar.set_label(r"$Corr(|X_c|, |X_d|)$")

    excess_image = excess_axis.imshow(
        excess_corr.to_numpy(),
        vmin=excess_vmin,
        vmax=excess_vmax,
        cmap=excess_cmap,
        aspect="equal",
    )
    excess_axis.set_title("Excess Correlation vs Shuffled Baseline")
    excess_colorbar = fig.colorbar(
        excess_image,
        ax=excess_axis,
        fraction=0.046,
        pad=0.04,
    )
    excess_colorbar.set_label(r"$\rho^{EURUSD}_{c,d} - \rho^{shuffle}_{c,d}$")

    for axis in axes:
        axis.set_xticks(np.arange(len(components)))
        axis.set_yticks(np.arange(len(components)))
        axis.set_xticklabels(components, rotation=45, ha="right")
        axis.set_yticklabels(components)

    fig.suptitle(f"Cross-Scale Volatility Coupling{title_suffix}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_memo_entropy_profile(
    entropy: pd.DataFrame,
    output_path: Path,
    k: int,
) -> Path:
    components = decomposition_components(k, include_original=False)
    theoretical_probabilities = np.array(
        [1 / 8, 3 / 16, 3 / 16, 3 / 16, 3 / 16, 1 / 8],
        dtype=float,
    )
    theoretical_entropy = float(
        -np.sum(theoretical_probabilities * np.log(theoretical_probabilities))
    )
    theoretical_normalized_entropy = theoretical_entropy / math.log(6)

    fig, axis = plt.subplots(figsize=(11, 5.8))
    for series, color, label in [
        (SERIES_FINAL, FINAL_COLOR, "EUR/USD"),
        (SERIES_SHUFFLE, SHUFFLE_COLOR, "Shuffled baseline"),
        (SERIES_GAUSSIAN, GAUSSIAN_COLOR, "Gaussian baseline"),
    ]:
        series_frame = (
            entropy[entropy["series"] == series]
            .set_index("component")
            .reindex(components)
        )
        axis.plot(
            components,
            series_frame["normalized_entropy"].astype(float).to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=color,
            label=label,
        )

    axis.axhline(
        theoretical_normalized_entropy,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label=f"Theoretical reference ({theoretical_normalized_entropy:.4f})",
    )
    axis.set_title("Normalized Permutation Entropy Across Scales")
    axis.set_xlabel("Component")
    axis.set_ylabel("Normalized entropy")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path
