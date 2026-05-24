"""In-process pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.baselines import BaselinePaths, create_baselines
from src.decomposition import DecompositionPaths, run_decomposition
from src.entropy import EntropyPaths, compute_entropy_metrics
from src.globals.constants import DEFAULT_K
from src.length_standardization import LengthStandardizationPaths, standardize_length
from src.plotting.decomposition import DecompositionPlotPaths, create_decomposition_plots
from src.plotting.eda import PlotPaths, create_eda_plots
from src.plotting.entropy import EntropyPlotPaths, create_entropy_plots
from src.plotting.memo import MemoPlotPaths, create_memo_plots
from src.plotting.volatility import VolatilityPlotPaths, create_volatility_plots
from src.preprocessing import PreprocessingPaths, run_preprocessing
from src.volatility import VolatilityPaths, compute_volatility_metrics


@dataclass(frozen=True)
class PipelineOptions:
    k: int = DEFAULT_K
    include_plots: bool = True


def run_core_pipeline(options: PipelineOptions | None = None) -> dict[str, Any]:
    options = options or PipelineOptions()
    return {
        "preprocessing": run_preprocessing(PreprocessingPaths()),
        "length_standardization": standardize_length(
            LengthStandardizationPaths(),
            k=options.k,
        ),
        "baselines": create_baselines(BaselinePaths()),
        "decomposition": run_decomposition(DecompositionPaths(), k=options.k),
        "volatility": compute_volatility_metrics(VolatilityPaths(), k=options.k),
        "entropy": compute_entropy_metrics(EntropyPaths(), k=options.k),
    }


def run_plot_pipeline(options: PipelineOptions | None = None) -> dict[str, list[Path]]:
    options = options or PipelineOptions()
    return {
        "eda_plots": create_eda_plots(PlotPaths()),
        "decomposition_plots": create_decomposition_plots(
            DecompositionPlotPaths(),
            k=options.k,
        ),
        "volatility_plots": create_volatility_plots(
            VolatilityPlotPaths(),
            k=options.k,
        ),
        "entropy_plots": create_entropy_plots(EntropyPlotPaths(), k=options.k),
        "memo_plots": create_memo_plots(MemoPlotPaths(), k=options.k),
    }


def run_all(options: PipelineOptions | None = None) -> dict[str, Any]:
    options = options or PipelineOptions()
    results = run_core_pipeline(options)
    if options.include_plots:
        results.update(run_plot_pipeline(options))
    return results

