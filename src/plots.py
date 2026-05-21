"""Compatibility facade for plotting APIs.

The plotting implementation lives in :mod:`src.plotting`. This module keeps
existing imports such as ``from src.plots import create_eda_plots`` working.
"""

from src.plotting.decomposition import (
    DecompositionPlotPaths,
    create_decomposition_plots,
)
from src.plotting.eda import PlotPaths, create_eda_plots
from src.plotting.entropy import EntropyPlotPaths, create_entropy_plots
from src.plotting.memo import MemoPlotPaths, create_memo_plots
from src.plotting.primitives import (
    plot_acf_comparison,
    plot_abs_component_correlation_heatmap,
    plot_abs_component_correlation_difference_heatmap,
    plot_decomposition_layers,
    plot_ecdf_comparison,
    plot_entropy_gaps,
    plot_entropy_metric,
    plot_entropy_pattern_distribution_grid,
    plot_histogram_comparison,
    plot_layer_acf_grid,
    plot_layer_histogram_grid,
    plot_layer_qq_grid,
    plot_qq_against_zero_mean_gaussian,
    plot_return_line,
    plot_volatility_difference_metric,
    plot_volatility_metric,
)
from src.plotting.volatility import VolatilityPlotPaths, create_volatility_plots

__all__ = [
    "DecompositionPlotPaths",
    "EntropyPlotPaths",
    "MemoPlotPaths",
    "PlotPaths",
    "VolatilityPlotPaths",
    "create_decomposition_plots",
    "create_eda_plots",
    "create_entropy_plots",
    "create_memo_plots",
    "create_volatility_plots",
    "plot_acf_comparison",
    "plot_abs_component_correlation_heatmap",
    "plot_abs_component_correlation_difference_heatmap",
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
    "plot_volatility_difference_metric",
    "plot_volatility_metric",
]

