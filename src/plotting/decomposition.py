"""High-level decomposition plot orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.constants import DEFAULT_K
from src.config.paths import (
    DECOMPOSITION_PLOTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    SHUFFLE_DECOMPOSITION_CSV,
)
from src.plotting.primitives import (
    plot_abs_component_correlation_heatmap,
    plot_abs_component_correlation_difference_heatmap,
    plot_decomposition_layers,
    plot_layer_acf_grid,
    plot_layer_histogram_grid,
    plot_layer_qq_grid,
)
from src.plotting.readers import read_decomposition
from src.utils.validation import require_positive_k


@dataclass(frozen=True)
class DecompositionPlotPaths:
    final_csv: Path = FINAL_DECOMPOSITION_CSV
    shuffle_csv: Path = SHUFFLE_DECOMPOSITION_CSV
    gaussian_csv: Path = GAUSSIAN_DECOMPOSITION_CSV
    output_dir: Path = DECOMPOSITION_PLOTS_DIR


def create_decomposition_plots(
    paths: DecompositionPlotPaths | None = None,
    k: int = DEFAULT_K,
    short_max_acf_lag: int = 1440,
    long_max_acf_lag: int = 6336,
) -> list[Path]:
    paths = paths or DecompositionPlotPaths()
    require_positive_k(k)
    if short_max_acf_lag < 1:
        raise ValueError("short_max_acf_lag must be at least 1")
    if long_max_acf_lag < 1:
        raise ValueError("long_max_acf_lag must be at least 1")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    final_frame = read_decomposition(paths.final_csv, k)
    shuffle_frame = read_decomposition(paths.shuffle_csv, k)
    gaussian_frame = read_decomposition(paths.gaussian_csv, k)

    return [
        plot_decomposition_layers(
            final_frame,
            paths.output_dir / "final_layers.png",
            title="EUR/USD Final Decomposition Layers",
            k=k,
        ),
        plot_decomposition_layers(
            shuffle_frame,
            paths.output_dir / "shuffle_layers.png",
            title="Shuffled Baseline Decomposition Layers",
            k=k,
        ),
        plot_decomposition_layers(
            gaussian_frame,
            paths.output_dir / "gaussian_layers.png",
            title="Gaussian Baseline Decomposition Layers",
            k=k,
        ),
        plot_layer_histogram_grid(
            final_frame,
            gaussian_frame,
            paths.output_dir / "layer_histograms_grid.png",
            k=k,
        ),
        plot_layer_qq_grid(
            final_frame,
            paths.output_dir / "layer_qq_gaussian_grid.png",
            k=k,
        ),
        plot_abs_component_correlation_heatmap(
            final_frame,
            paths.output_dir / "final_abs_component_correlation.png",
            title="EUR/USD Final Absolute Component Correlation",
            k=k,
        ),
        plot_abs_component_correlation_heatmap(
            shuffle_frame,
            paths.output_dir / "shuffle_abs_component_correlation.png",
            title="Shuffled Baseline Absolute Component Correlation",
            k=k,
        ),
        plot_abs_component_correlation_heatmap(
            gaussian_frame,
            paths.output_dir / "gaussian_abs_component_correlation.png",
            title="Gaussian Baseline Absolute Component Correlation",
            k=k,
        ),
        plot_abs_component_correlation_difference_heatmap(
            final_frame,
            shuffle_frame,
            paths.output_dir / "final_minus_shuffle_abs_component_correlation.png",
            title="Absolute Component Correlation Difference: Final Minus Shuffled",
            k=k,
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_returns_short_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(1, min(k, 6) + 1)],
            max_lag=short_max_acf_lag,
            absolute=False,
            title="Short-Scale Layer Autocorrelation",
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_abs_returns_short_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(1, min(k, 6) + 1)],
            max_lag=short_max_acf_lag,
            absolute=True,
            title="Short-Scale Absolute Layer Autocorrelation",
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_returns_long_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(7, k + 1)] + [f"A_{k:02d}"],
            max_lag=long_max_acf_lag,
            absolute=False,
            title="Long-Scale Layer Autocorrelation",
        ),
        plot_layer_acf_grid(
            final_frame,
            shuffle_frame,
            gaussian_frame,
            paths.output_dir / "layer_acf_abs_returns_long_scales.png",
            layers=[f"D_{scale:02d}" for scale in range(7, k + 1)] + [f"A_{k:02d}"],
            max_lag=long_max_acf_lag,
            absolute=True,
            title="Long-Scale Absolute Layer Autocorrelation",
        ),
    ]
