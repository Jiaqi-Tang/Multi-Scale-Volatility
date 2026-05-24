"""Reusable matplotlib plotting primitives."""

from src.plotting.primitives.acf import plot_acf_comparison, plot_layer_acf_grid
from src.plotting.primitives.distributions import (
    add_mean_median_lines,
    plot_ecdf_comparison,
    plot_histogram_comparison,
)
from src.plotting.primitives.grids import (
    plot_entropy_pattern_distribution_grid,
    plot_layer_histogram_grid,
    plot_layer_qq_grid,
)
from src.plotting.primitives.heatmaps import (
    plot_abs_component_correlation_difference_heatmap,
    plot_abs_component_correlation_heatmap,
)
from src.plotting.primitives.lines import plot_decomposition_layers, plot_return_line
from src.plotting.primitives.metrics import (
    metric_components,
    plot_baseline_difference_metric,
    plot_entropy_gaps,
    plot_entropy_metric,
    plot_series_metric,
    plot_volatility_difference_metric,
    plot_volatility_metric,
)
from src.plotting.primitives.qq import plot_qq_against_zero_mean_gaussian

__all__ = [
    "add_mean_median_lines",
    "metric_components",
    "plot_abs_component_correlation_difference_heatmap",
    "plot_abs_component_correlation_heatmap",
    "plot_acf_comparison",
    "plot_baseline_difference_metric",
    "plot_decomposition_layers",
    "plot_ecdf_comparison",
    "plot_entropy_gaps",
    "plot_entropy_metric",
    "plot_entropy_pattern_distribution_grid",
    "plot_histogram_comparison",
    "plot_layer_acf_grid",
    "plot_layer_histogram_grid",
    "plot_layer_qq_grid",
    "plot_qq_against_zero_mean_gaussian",
    "plot_return_line",
    "plot_series_metric",
    "plot_volatility_difference_metric",
    "plot_volatility_metric",
]

