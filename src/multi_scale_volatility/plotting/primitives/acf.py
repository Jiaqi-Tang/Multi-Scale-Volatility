"""Autocorrelation plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.plotting.style import (
    FIGURE_DPI,
    FINAL_COLOR,
    GAUSSIAN_COLOR,
    GRID_FIGURE_DPI,
    SHUFFLE_COLOR,
)
from multi_scale_volatility.scale_utils import compress_component
from multi_scale_volatility.stats import autocorrelation, compressed_layer_autocorrelation


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
    ax.plot(lags, final_acf, linewidth=1.2,
            label="EUR/USD final", color=FINAL_COLOR)
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
    fig.suptitle(
        f"{title} (compressed repeated block values for plotting)", fontsize=16)

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
        axis.plot(shuffle_lags, shuffle_acf, linewidth=0.9,
                  label="Shuffled", alpha=0.9)
        axis.plot(gaussian_lags, gaussian_acf,
                  linewidth=0.9, label="Gaussian", alpha=0.9)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.75)
        axis.axhline(band, color="black", linestyle="--",
                     linewidth=0.7, alpha=0.45)
        axis.axhline(-band, color="black", linestyle="--",
                     linewidth=0.7, alpha=0.45)
        axis.set_ylabel(layer)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Lag")
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
    plt.close(fig)
    return output_path
