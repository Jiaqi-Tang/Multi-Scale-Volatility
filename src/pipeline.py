"""In-process pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.baselines import BaselinePaths, create_baselines
from src.decomposition import DecompositionPaths, run_decomposition
from src.entropy import EntropyPaths, compute_entropy_metrics
from src.config.constants import DEFAULT_K
from src.length_standardization import LengthStandardizationPaths, standardize_length
from src.plotting.decomposition import DecompositionPlotPaths, create_decomposition_plots
from src.plotting.eda import PlotPaths, create_eda_plots
from src.plotting.entropy import EntropyPlotPaths, create_entropy_plots
from src.plotting.memo import MemoPlotPaths, create_memo_plots
from src.plotting.volatility import VolatilityPlotPaths, create_volatility_plots
from src.preprocessing import PreprocessingPaths, run_preprocessing
from src.volatility import VolatilityPaths, compute_volatility_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    k: int = DEFAULT_K
    include_plots: bool = True


def run_core_pipeline(options: PipelineOptions | None = None) -> dict[str, Any]:
    options = options or PipelineOptions()
    results: dict[str, Any] = {}
    stages = [
        ("preprocessing", lambda: run_preprocessing(PreprocessingPaths())),
        (
            "length_standardization",
            lambda: standardize_length(LengthStandardizationPaths(), k=options.k),
        ),
        ("baselines", lambda: create_baselines(BaselinePaths())),
        ("decomposition", lambda: run_decomposition(DecompositionPaths(), k=options.k)),
        ("volatility", lambda: compute_volatility_metrics(VolatilityPaths(), k=options.k)),
        ("entropy", lambda: compute_entropy_metrics(EntropyPaths(), k=options.k)),
    ]
    for name, run_stage in stages:
        logger.info("Starting %s", name)
        results[name] = run_stage()
        logger.info("Finished %s", name)
    return results


def run_plot_pipeline(options: PipelineOptions | None = None) -> dict[str, list[Path]]:
    options = options or PipelineOptions()
    results: dict[str, list[Path]] = {}
    stages = [
        ("eda_plots", lambda: create_eda_plots(PlotPaths())),
        (
            "decomposition_plots",
            lambda: create_decomposition_plots(DecompositionPlotPaths(), k=options.k),
        ),
        (
            "volatility_plots",
            lambda: create_volatility_plots(VolatilityPlotPaths(), k=options.k),
        ),
        ("entropy_plots", lambda: create_entropy_plots(EntropyPlotPaths(), k=options.k)),
        ("memo_plots", lambda: create_memo_plots(MemoPlotPaths(), k=options.k)),
    ]
    for name, run_stage in stages:
        logger.info("Starting %s", name)
        results[name] = run_stage()
        logger.info("Finished %s", name)
    return results


def run_all(options: PipelineOptions | None = None) -> dict[str, Any]:
    options = options or PipelineOptions()
    results = run_core_pipeline(options)
    if options.include_plots:
        results.update(run_plot_pipeline(options))
    return results
