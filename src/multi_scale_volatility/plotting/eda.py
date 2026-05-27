"""High-level EDA plot orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multi_scale_volatility.config.paths import (
    EDA_PLOTS_DIR,
    FINAL_RETURNS_CSV,
    GAUSSIAN_RETURNS_CSV,
    SHUFFLE_RETURNS_CSV,
)
from multi_scale_volatility.plotting.primitives import (
    plot_acf_comparison,
    plot_ecdf_comparison,
    plot_histogram_comparison,
    plot_qq_against_zero_mean_gaussian,
    plot_return_line,
)
from multi_scale_volatility.plotting.readers import read_returns


@dataclass(frozen=True)
class EdaPlotPaths:
    final_csv: Path = FINAL_RETURNS_CSV
    shuffle_csv: Path = SHUFFLE_RETURNS_CSV
    gaussian_csv: Path = GAUSSIAN_RETURNS_CSV
    output_dir: Path = EDA_PLOTS_DIR


def create_eda_plots(
    paths: EdaPlotPaths | None = None,
    max_acf_lag: int = 288,
) -> list[Path]:
    paths = paths or EdaPlotPaths()
    if max_acf_lag < 1:
        raise ValueError("max_acf_lag must be at least 1")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    final_returns = read_returns(paths.final_csv)
    shuffle_returns = read_returns(paths.shuffle_csv)
    gaussian_returns = read_returns(paths.gaussian_csv)

    return [
        plot_return_line(
            final_returns,
            paths.output_dir / "final_returns_line.png",
            title="EUR/USD Final 5m Log Returns",
        ),
        plot_return_line(
            shuffle_returns,
            paths.output_dir / "shuffle_returns_line.png",
            title="Shuffled Baseline 5m Log Returns",
        ),
        plot_return_line(
            gaussian_returns,
            paths.output_dir / "gaussian_returns_line.png",
            title="Gaussian Baseline 5m Log Returns",
        ),
        plot_histogram_comparison(
            final_returns,
            gaussian_returns,
            paths.output_dir / "final_vs_gaussian_histogram.png",
        ),
        plot_ecdf_comparison(
            final_returns,
            gaussian_returns,
            paths.output_dir / "final_vs_gaussian_ecdf.png",
        ),
        plot_qq_against_zero_mean_gaussian(
            final_returns,
            paths.output_dir / "final_qq_gaussian.png",
        ),
        plot_acf_comparison(
            final_returns,
            shuffle_returns,
            gaussian_returns,
            paths.output_dir / "final_vs_baselines_returns_acf.png",
            max_lag=max_acf_lag,
            title="Autocorrelation of 5m Log Returns",
            transform_label="returns",
        ),
        plot_acf_comparison(
            np.abs(final_returns),
            np.abs(shuffle_returns),
            np.abs(gaussian_returns),
            paths.output_dir / "final_vs_baselines_abs_returns_acf.png",
            max_lag=max_acf_lag,
            title="Autocorrelation of Absolute 5m Log Returns",
            transform_label="absolute returns",
        ),
    ]
