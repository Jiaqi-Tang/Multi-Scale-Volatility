"""In-process pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from multi_scale_volatility.research.global_diagnosis.baselines import BaselinePaths, create_baselines
from multi_scale_volatility.core.config.names import SERIES
from multi_scale_volatility.core.config.names import BASE_INTERVAL_MINUTES, DEFAULT_K
from multi_scale_volatility.core.config.paths import (
    DECOMPOSITION_DIR,
    DECOMPOSITION_REPORT_JSON,
    ENTROPY_REPORT_JSON,
    ENTROPY_RESULTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    LAYER_ENTROPY_CSV,
    VOLATILITY_CSV,
    VOLATILITY_REPORT_JSON,
    VOLATILITY_RESULTS_DIR,
)
from multi_scale_volatility.core.config.names import SERIES_FINAL
from multi_scale_volatility.research.decomposition import DecompositionInput, decompose_csv
from multi_scale_volatility.research.length_standardization import LengthStandardizationPaths, standardize_length
from multi_scale_volatility.research.global_diagnosis.monte_carlo_metrics import (
    MonteCarloMetricPaths,
    compute_entropy_rows,
    compute_monte_carlo_metrics,
    compute_volatility_rows,
)
from multi_scale_volatility.plotting.global_results import (
    MonteCarloBaselinePlotPaths,
    create_v11_memo_plots,
)
from multi_scale_volatility.research.preprocessing import PreprocessingPaths, run_preprocessing
from multi_scale_volatility.core.io import write_csv
from multi_scale_volatility.core.io import write_json
from multi_scale_volatility.app.runtime import get_logger, logged_stage

logger = get_logger(__name__)


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
            lambda: standardize_length(
                LengthStandardizationPaths(), k=options.k),
        ),
        ("empirical_decomposition", lambda: run_empirical_decomposition(k=options.k)),
        ("empirical_metrics", lambda: compute_empirical_metrics(k=options.k)),
        ("monte_carlo_baselines", lambda: create_baselines(BaselinePaths(), k=options.k)),
        (
            "monte_carlo_metrics",
            lambda: compute_monte_carlo_metrics(MonteCarloMetricPaths(), k=options.k),
        ),
    ]
    for name, run_stage in stages:
        with logged_stage(logger, name):
            results[name] = run_stage()
    return results


def run_plot_pipeline(options: PipelineOptions | None = None) -> dict[str, list[Path]]:
    options = options or PipelineOptions()
    results: dict[str, list[Path]] = {}
    stages = [
        (
            "v11_plots",
            lambda: create_v11_memo_plots(
                MonteCarloBaselinePlotPaths(),
                k=options.k,
            ),
        ),
    ]
    for name, run_stage in stages:
        with logged_stage(logger, name):
            results[name] = run_stage()
    return results


def run_all(options: PipelineOptions | None = None) -> dict[str, Any]:
    options = options or PipelineOptions()
    results = run_core_pipeline(options)
    if options.include_plots:
        results.update(run_plot_pipeline(options))
    return results


def run_empirical_decomposition(k: int = DEFAULT_K) -> dict[str, Any]:
    DECOMPOSITION_DIR.mkdir(parents=True, exist_ok=True)
    report = decompose_csv(
        DecompositionInput(
            name=SERIES_FINAL,
            input_csv=FINAL_RETURNS_CSV,
            output_csv=FINAL_DECOMPOSITION_CSV,
        ),
        k=k,
    )
    output = {
        "K": k,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "series": {SERIES_FINAL: report},
    }
    write_json(DECOMPOSITION_REPORT_JSON, output)
    return output


def compute_empirical_metrics(k: int = DEFAULT_K) -> dict[str, Any]:
    frame = pd.read_csv(FINAL_DECOMPOSITION_CSV)

    volatility_rows = [
        empirical_metric_row(row)
        for row in compute_volatility_rows(
            frame,
            baseline_type=SERIES_FINAL,
            simulation_id=0,
            k=k,
        )
    ]
    VOLATILITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(volatility_rows), VOLATILITY_CSV, index=False)
    write_json(
        VOLATILITY_REPORT_JSON,
        {
            "K": k,
            "base_interval_minutes": BASE_INTERVAL_MINUTES,
            "output_csv": str(VOLATILITY_CSV),
            "series": [SERIES_FINAL],
        },
    )

    entropy_rows = [
        empirical_metric_row(row)
        for row in compute_entropy_rows(
            frame,
            baseline_type=SERIES_FINAL,
            simulation_id=0,
            k=k,
            embedding_dimension=3,
            delay=1,
            jitter_seed=314,
            jitter_magnitude=1e-10,
        )
    ]
    ENTROPY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(entropy_rows), LAYER_ENTROPY_CSV, index=False)
    write_json(
        ENTROPY_REPORT_JSON,
        {
            "K": k,
            "base_interval_minutes": BASE_INTERVAL_MINUTES,
            "layer_entropy_csv": str(LAYER_ENTROPY_CSV),
            "series": [SERIES_FINAL],
        },
    )

    return {
        "volatility_csv": str(VOLATILITY_CSV),
        "entropy_csv": str(LAYER_ENTROPY_CSV),
    }


def empirical_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output[SERIES] = SERIES_FINAL
    output.pop("baseline_type", None)
    output.pop("simulation_id", None)
    return output
