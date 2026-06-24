"""Metric tables and baseline envelopes for Monte Carlo baseline simulations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.components import ComponentSpec, component_specs
from multi_scale_volatility.config.names import (
    COMPONENT,
    COMPONENT_TYPE,
    LOG_RETURN,
    ORIGINAL,
    TIMESTAMP_UTC,
)
from multi_scale_volatility.config.names import (
    BASE_INTERVAL_MINUTES,
    DEFAULT_K,
    MONTE_CARLO_BASELINE_QUANTILE_METHOD,
)
from multi_scale_volatility.config.names import (
    ANNUALIZED_RMS_VOLATILITY,
    DETAIL_ENERGY_SHARE,
    EFFECTIVE_N,
    ENERGY,
    K,
    NORMALIZED_ENTROPY,
    ORDINAL_WINDOWS,
    PERMUTATION_ENTROPY,
    REPEAT_LENGTH,
    RMS_VOLATILITY,
    SCALE_DAYS,
    SCALE_MINUTES,
    TOTAL_COMPONENT_ENERGY_SHARE,
)
from multi_scale_volatility.config.paths import (
    MC_ABS_COMPONENT_CORRELATION_SIMULATIONS_CSV,
    MC_ABS_COMPONENT_CORRELATION_SUMMARY_CSV,
    MC_ABS_COMPONENT_CORRELATION_EMPIRICAL_COMPARISON_CSV,
    MC_ACF_SIMULATIONS_CSV,
    MC_ACF_SUMMARY_CSV,
    MC_ACF_EMPIRICAL_COMPARISON_CSV,
    MC_COMPONENT_ACF_EMPIRICAL_COMPARISON_CSV,
    MC_COMPONENT_ACF_SIMULATIONS_CSV,
    MC_COMPONENT_ACF_SUMMARY_CSV,
    MC_LAYER_ENTROPY_EMPIRICAL_COMPARISON_CSV,
    MC_LAYER_ENTROPY_SIMULATIONS_CSV,
    MC_LAYER_ENTROPY_SUMMARY_CSV,
    MC_LAYER_VOLATILITY_EMPIRICAL_COMPARISON_CSV,
    MC_LAYER_VOLATILITY_SIMULATIONS_CSV,
    MC_LAYER_VOLATILITY_SUMMARY_CSV,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    LAYER_ENTROPY_CSV,
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV,
    MONTE_CARLO_BASELINES_RESULTS_DIR,
    VOLATILITY_CSV,
)
from multi_scale_volatility.config.names import SERIES_FINAL
from multi_scale_volatility.entropy import (
    DELAY,
    EMBEDDING_DIMENSION,
    JITTER_MAGNITUDE,
    JITTER_SEED,
    _add_jitter,
    _component_jitter_seed,
    _permutation_entropy,
)
from multi_scale_volatility.components import compress_component, decomposition_components
from multi_scale_volatility.stats import (
    absolute_component_correlation,
    autocorrelation,
    compressed_layer_autocorrelation,
)
from multi_scale_volatility.io import write_csv
from multi_scale_volatility.runtime import (
    RuntimeTracker,
    get_logger,
    runtime_row,
    start_timer,
)
from multi_scale_volatility.utils.validation import require_finite_array, require_positive_k

RETURN_ACF_MAX_LAG = 288
ABS_RETURN_ACF_MAX_LAG = 1440
SHORT_COMPONENT_ACF_MAX_LAG = 1440
LONG_COMPONENT_ACF_MAX_LAG = 6336
SUMMARY_QUANTILES = (0.05, 0.5, 0.95)
RUNTIME_LOG_BATCH_SIZE = 10
logger = get_logger(__name__)


@dataclass(frozen=True)
class MonteCarloMetricPaths:
    audit_csv: Path = MONTE_CARLO_BASELINE_AUDIT_CSV
    results_dir: Path = MONTE_CARLO_BASELINES_RESULTS_DIR
    runtime_log_csv: Path = MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV
    final_returns_csv: Path = FINAL_RETURNS_CSV
    final_decomposition_csv: Path = FINAL_DECOMPOSITION_CSV
    empirical_volatility_csv: Path = VOLATILITY_CSV
    empirical_entropy_csv: Path = LAYER_ENTROPY_CSV

    @property
    def layer_volatility_simulations_csv(self) -> Path:
        return self.results_dir / MC_LAYER_VOLATILITY_SIMULATIONS_CSV.name

    @property
    def layer_volatility_summary_csv(self) -> Path:
        return self.results_dir / MC_LAYER_VOLATILITY_SUMMARY_CSV.name

    @property
    def layer_volatility_empirical_comparison_csv(self) -> Path:
        return self.results_dir / MC_LAYER_VOLATILITY_EMPIRICAL_COMPARISON_CSV.name

    @property
    def layer_entropy_simulations_csv(self) -> Path:
        return self.results_dir / MC_LAYER_ENTROPY_SIMULATIONS_CSV.name

    @property
    def layer_entropy_summary_csv(self) -> Path:
        return self.results_dir / MC_LAYER_ENTROPY_SUMMARY_CSV.name

    @property
    def layer_entropy_empirical_comparison_csv(self) -> Path:
        return self.results_dir / MC_LAYER_ENTROPY_EMPIRICAL_COMPARISON_CSV.name

    @property
    def acf_simulations_csv(self) -> Path:
        return self.results_dir / MC_ACF_SIMULATIONS_CSV.name

    @property
    def acf_summary_csv(self) -> Path:
        return self.results_dir / MC_ACF_SUMMARY_CSV.name

    @property
    def acf_empirical_comparison_csv(self) -> Path:
        return self.results_dir / MC_ACF_EMPIRICAL_COMPARISON_CSV.name

    @property
    def component_acf_simulations_csv(self) -> Path:
        return self.results_dir / MC_COMPONENT_ACF_SIMULATIONS_CSV.name

    @property
    def component_acf_summary_csv(self) -> Path:
        return self.results_dir / MC_COMPONENT_ACF_SUMMARY_CSV.name

    @property
    def component_acf_empirical_comparison_csv(self) -> Path:
        return self.results_dir / MC_COMPONENT_ACF_EMPIRICAL_COMPARISON_CSV.name

    @property
    def abs_component_correlation_simulations_csv(self) -> Path:
        return self.results_dir / MC_ABS_COMPONENT_CORRELATION_SIMULATIONS_CSV.name

    @property
    def abs_component_correlation_summary_csv(self) -> Path:
        return self.results_dir / MC_ABS_COMPONENT_CORRELATION_SUMMARY_CSV.name

    @property
    def abs_component_correlation_empirical_comparison_csv(self) -> Path:
        return self.results_dir / MC_ABS_COMPONENT_CORRELATION_EMPIRICAL_COMPARISON_CSV.name


@dataclass(frozen=True)
class SimulationMetricResult:
    volatility_rows: list[dict[str, Any]]
    entropy_rows: list[dict[str, Any]]
    acf_rows: list[dict[str, Any]]
    component_acf_rows: list[dict[str, Any]]
    corr_rows: list[dict[str, Any]]
    runtime_rows: list[dict[str, Any]]
    status: str
    error_message: str


def _compute_simulation_metrics_worker(args: tuple[Any, ...]) -> SimulationMetricResult:
    (
        record,
        specs,
        components,
        k,
        embedding_dimension,
        delay,
        jitter_seed,
        jitter_magnitude,
        return_acf_max_lag,
        abs_return_acf_max_lag,
        short_component_acf_max_lag,
        long_component_acf_max_lag,
    ) = args
    baseline_type = str(record["baseline_type"])
    simulation_id = int(record["simulation_id"])
    return_parquet = record["return_parquet"]
    decomposition_parquet = record["decomposition_parquet"]
    n = int(record["n"])

    simulation_timer = start_timer()
    runtime_rows: list[dict[str, Any]] = []
    rows_out = 0
    try:
        read_timer = start_timer()
        returns = pd.read_parquet(return_parquet, columns=[LOG_RETURN])
        decomposition = pd.read_parquet(decomposition_parquet, columns=components)
        values = returns[LOG_RETURN].astype(float).to_numpy()
        require_finite_array(values, f"{baseline_type} {simulation_id} returns")
        runtime_rows.append(
            runtime_row(
                run_id="",
                stage="monte_carlo_metrics",
                operation="read_simulation_artifacts",
                started_at_utc=read_timer.started_at_utc,
                elapsed_seconds=read_timer.elapsed_seconds,
                baseline_type=baseline_type,
                simulation_id=simulation_id,
                status="success",
                rows_in=n,
                rows_out=len(decomposition),
                output_path=f"{return_parquet};{decomposition_parquet}",
                error_message="",
            )
        )

        metric_timer = start_timer()
        volatility_rows = compute_volatility_rows(
            decomposition,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            k=k,
            specs=specs,
        )
        rows_out += len(volatility_rows)
        runtime_rows.append(
            runtime_row(
                run_id="",
                stage="monte_carlo_metrics",
                operation="compute_volatility_metrics",
                started_at_utc=metric_timer.started_at_utc,
                elapsed_seconds=metric_timer.elapsed_seconds,
                baseline_type=baseline_type,
                simulation_id=simulation_id,
                status="success",
                rows_in=len(decomposition),
                rows_out=len(volatility_rows),
                output_path=str(decomposition_parquet),
                error_message="",
            )
        )

        metric_timer = start_timer()
        entropy_rows = compute_entropy_rows(
            decomposition,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            k=k,
            embedding_dimension=embedding_dimension,
            delay=delay,
            jitter_seed=jitter_seed,
            jitter_magnitude=jitter_magnitude,
            specs=specs,
        )
        rows_out += len(entropy_rows)
        runtime_rows.append(
            runtime_row(
                run_id="",
                stage="monte_carlo_metrics",
                operation="compute_entropy_metrics",
                started_at_utc=metric_timer.started_at_utc,
                elapsed_seconds=metric_timer.elapsed_seconds,
                baseline_type=baseline_type,
                simulation_id=simulation_id,
                status="success",
                rows_in=len(decomposition),
                rows_out=len(entropy_rows),
                output_path=str(decomposition_parquet),
                error_message="",
            )
        )

        metric_timer = start_timer()
        acf_rows = compute_acf_rows(
            values,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            return_acf_max_lag=return_acf_max_lag,
            abs_return_acf_max_lag=abs_return_acf_max_lag,
        )
        rows_out += len(acf_rows)
        runtime_rows.append(
            runtime_row(
                run_id="",
                stage="monte_carlo_metrics",
                operation="compute_acf_metrics",
                started_at_utc=metric_timer.started_at_utc,
                elapsed_seconds=metric_timer.elapsed_seconds,
                baseline_type=baseline_type,
                simulation_id=simulation_id,
                status="success",
                rows_in=len(values),
                rows_out=len(acf_rows),
                output_path=str(return_parquet),
                error_message="",
            )
        )

        metric_timer = start_timer()
        component_acf_rows = compute_component_acf_rows(
            decomposition,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            k=k,
            short_max_lag=short_component_acf_max_lag,
            long_max_lag=long_component_acf_max_lag,
            specs=specs,
        )
        rows_out += len(component_acf_rows)
        runtime_rows.append(
            runtime_row(
                run_id="",
                stage="monte_carlo_metrics",
                operation="compute_component_acf_metrics",
                started_at_utc=metric_timer.started_at_utc,
                elapsed_seconds=metric_timer.elapsed_seconds,
                baseline_type=baseline_type,
                simulation_id=simulation_id,
                status="success",
                rows_in=len(decomposition),
                rows_out=len(component_acf_rows),
                output_path=str(decomposition_parquet),
                error_message="",
            )
        )

        metric_timer = start_timer()
        corr_rows = compute_abs_component_correlation_rows(
            decomposition,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            k=k,
            components=components,
        )
        rows_out += len(corr_rows)
        runtime_rows.append(
            runtime_row(
                run_id="",
                stage="monte_carlo_metrics",
                operation="compute_abs_component_correlation_metrics",
                started_at_utc=metric_timer.started_at_utc,
                elapsed_seconds=metric_timer.elapsed_seconds,
                baseline_type=baseline_type,
                simulation_id=simulation_id,
                status="success",
                rows_in=len(decomposition),
                rows_out=len(corr_rows),
                output_path=str(decomposition_parquet),
                error_message="",
            )
        )
        status = "success"
        error_message = ""
    except Exception as error:
        status = "error"
        error_message = str(error)
        volatility_rows = []
        entropy_rows = []
        acf_rows = []
        component_acf_rows = []
        corr_rows = []

    runtime_rows.append(
        runtime_row(
            run_id="",
            stage="monte_carlo_metrics",
            operation="compute_simulation_metrics",
            started_at_utc=simulation_timer.started_at_utc,
            elapsed_seconds=simulation_timer.elapsed_seconds,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            status=status,
            rows_in=n,
            rows_out=rows_out if status == "success" else 0,
            output_path=str(decomposition_parquet),
            error_message=error_message,
        )
    )
    return SimulationMetricResult(
        volatility_rows=volatility_rows,
        entropy_rows=entropy_rows,
        acf_rows=acf_rows,
        component_acf_rows=component_acf_rows,
        corr_rows=corr_rows,
        runtime_rows=runtime_rows,
        status=status,
        error_message=error_message,
    )


def compute_monte_carlo_metrics(
    paths: MonteCarloMetricPaths | None = None,
    k: int = DEFAULT_K,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    delay: int = DELAY,
    jitter_seed: int = JITTER_SEED,
    jitter_magnitude: float = JITTER_MAGNITUDE,
    return_acf_max_lag: int = RETURN_ACF_MAX_LAG,
    abs_return_acf_max_lag: int = ABS_RETURN_ACF_MAX_LAG,
    short_component_acf_max_lag: int = SHORT_COMPONENT_ACF_MAX_LAG,
    long_component_acf_max_lag: int = LONG_COMPONENT_ACF_MAX_LAG,
    max_workers: int | None = None,
) -> dict[str, Any]:
    paths = paths or MonteCarloMetricPaths()
    require_positive_k(k)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(paths.audit_csv)
    if audit.empty:
        raise ValueError(f"Audit table is empty: {paths.audit_csv}")

    tracker = RuntimeTracker(paths.runtime_log_csv, flush_every=RUNTIME_LOG_BATCH_SIZE)
    stage_timer = start_timer()

    volatility_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    acf_rows: list[dict[str, Any]] = []
    component_acf_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []

    specs = component_specs(k, include_original=False, base_interval_minutes=BASE_INTERVAL_MINUTES)
    components = [spec.name for spec in specs]
    worker_args = [
        (
            record,
            specs,
            components,
            k,
            embedding_dimension,
            delay,
            jitter_seed,
            jitter_magnitude,
            return_acf_max_lag,
            abs_return_acf_max_lag,
            short_component_acf_max_lag,
            long_component_acf_max_lag,
        )
        for record in audit.to_dict("records")
    ]

    effective_max_workers = max_workers or min(4, os.cpu_count() or 1)
    logger.info(
        "Computing Monte Carlo metrics for %s simulations with %s workers",
        len(worker_args),
        effective_max_workers,
    )
    with ProcessPoolExecutor(max_workers=effective_max_workers) as executor:
        for index, result in enumerate(
            executor.map(_compute_simulation_metrics_worker, worker_args),
            start=1,
        ):
            volatility_rows.extend(result.volatility_rows)
            entropy_rows.extend(result.entropy_rows)
            acf_rows.extend(result.acf_rows)
            component_acf_rows.extend(result.component_acf_rows)
            corr_rows.extend(result.corr_rows)
            tracker.extend(
                result.runtime_rows,
                flush=index % RUNTIME_LOG_BATCH_SIZE == 0 or result.status != "success",
            )
            if result.status != "success":
                raise RuntimeError(result.error_message)
            if index % RUNTIME_LOG_BATCH_SIZE == 0:
                logger.info("Computed metrics for %s/%s simulations", index, len(worker_args))
    tracker.flush()

    volatility = pd.DataFrame(volatility_rows)
    entropy = pd.DataFrame(entropy_rows)
    acf = pd.DataFrame(acf_rows)
    component_acf = pd.DataFrame(component_acf_rows)
    corr = pd.DataFrame(corr_rows)

    write_csv(volatility, paths.layer_volatility_simulations_csv, index=False)
    write_csv(entropy, paths.layer_entropy_simulations_csv, index=False)
    write_csv(acf, paths.acf_simulations_csv, index=False)
    write_csv(component_acf, paths.component_acf_simulations_csv, index=False)
    write_csv(corr, paths.abs_component_correlation_simulations_csv, index=False)

    volatility_summary = summarize_metrics(
        volatility,
        group_cols=[
            "baseline_type",
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
        ],
        value_cols=[
            ENERGY,
            RMS_VOLATILITY,
            ANNUALIZED_RMS_VOLATILITY,
            DETAIL_ENERGY_SHARE,
            TOTAL_COMPONENT_ENERGY_SHARE,
        ],
    )
    entropy_summary = summarize_metrics(
        entropy,
        group_cols=[
            "baseline_type",
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
            REPEAT_LENGTH,
        ],
        value_cols=[
            EFFECTIVE_N,
            ORDINAL_WINDOWS,
            PERMUTATION_ENTROPY,
            NORMALIZED_ENTROPY,
        ],
    )
    acf_summary = summarize_metrics(
        acf,
        group_cols=["baseline_type", "acf_kind", "lag", "max_lag"],
        value_cols=["acf"],
    )
    component_acf_summary = summarize_metrics(
        component_acf,
        group_cols=[
            "baseline_type",
            COMPONENT,
            COMPONENT_TYPE,
            K,
            SCALE_MINUTES,
            SCALE_DAYS,
            "acf_kind",
            "lag",
            "compressed_lag",
            "max_lag",
        ],
        value_cols=["acf"],
    )
    corr_summary = summarize_metrics(
        corr,
        group_cols=["baseline_type", "component_i", "component_j"],
        value_cols=["correlation_abs"],
    )

    write_csv(volatility_summary, paths.layer_volatility_summary_csv, index=False)
    write_csv(entropy_summary, paths.layer_entropy_summary_csv, index=False)
    write_csv(acf_summary, paths.acf_summary_csv, index=False)
    write_csv(component_acf_summary, paths.component_acf_summary_csv, index=False)
    write_csv(corr_summary, paths.abs_component_correlation_summary_csv, index=False)

    volatility_comparison = compare_empirical_volatility(
        paths.empirical_volatility_csv,
        volatility,
        volatility_summary,
        k=k,
    )
    entropy_comparison = compare_empirical_entropy(
        paths.empirical_entropy_csv,
        entropy,
        entropy_summary,
        k=k,
    )
    acf_comparison = compare_empirical_acf(
        paths.final_returns_csv,
        acf,
        acf_summary,
        return_acf_max_lag=return_acf_max_lag,
        abs_return_acf_max_lag=abs_return_acf_max_lag,
    )
    component_acf_comparison = compare_empirical_component_acf(
        paths.final_decomposition_csv,
        component_acf,
        component_acf_summary,
        k=k,
        short_max_lag=short_component_acf_max_lag,
        long_max_lag=long_component_acf_max_lag,
    )
    corr_comparison = compare_empirical_abs_component_correlation(
        paths.final_decomposition_csv,
        corr,
        corr_summary,
        k=k,
    )
    write_csv(
        volatility_comparison,
        paths.layer_volatility_empirical_comparison_csv,
        index=False,
    )
    write_csv(
        entropy_comparison,
        paths.layer_entropy_empirical_comparison_csv,
        index=False,
    )
    write_csv(acf_comparison, paths.acf_empirical_comparison_csv, index=False)
    write_csv(
        component_acf_comparison,
        paths.component_acf_empirical_comparison_csv,
        index=False,
    )
    write_csv(
        corr_comparison,
        paths.abs_component_correlation_empirical_comparison_csv,
        index=False,
    )

    tracker.record(
        stage="monte_carlo_metrics",
        operation="compute_monte_carlo_metrics",
        started_at_utc=stage_timer.started_at_utc,
        elapsed_seconds=stage_timer.elapsed_seconds,
        rows_in=len(audit),
        rows_out=(
            len(volatility)
            + len(entropy)
            + len(acf)
            + len(component_acf)
            + len(corr)
            + len(volatility_summary)
            + len(entropy_summary)
            + len(acf_summary)
            + len(component_acf_summary)
            + len(corr_summary)
            + len(volatility_comparison)
            + len(entropy_comparison)
            + len(acf_comparison)
            + len(component_acf_comparison)
            + len(corr_comparison)
        ),
        output_path=str(paths.results_dir),
        flush=True,
    )
    logger.info("Wrote Monte Carlo metric outputs to %s", paths.results_dir)

    return {
        "input_audit_csv": str(paths.audit_csv),
        "simulation_count": int(len(audit)),
        "outputs": {
            "layer_volatility_simulations_csv": str(paths.layer_volatility_simulations_csv),
            "layer_volatility_summary_csv": str(paths.layer_volatility_summary_csv),
            "layer_entropy_simulations_csv": str(paths.layer_entropy_simulations_csv),
            "layer_entropy_summary_csv": str(paths.layer_entropy_summary_csv),
            "acf_simulations_csv": str(paths.acf_simulations_csv),
            "acf_summary_csv": str(paths.acf_summary_csv),
            "component_acf_simulations_csv": str(paths.component_acf_simulations_csv),
            "component_acf_summary_csv": str(paths.component_acf_summary_csv),
            "abs_component_correlation_simulations_csv": str(
                paths.abs_component_correlation_simulations_csv
            ),
            "abs_component_correlation_summary_csv": str(
                paths.abs_component_correlation_summary_csv
            ),
            "layer_volatility_empirical_comparison_csv": str(
                paths.layer_volatility_empirical_comparison_csv
            ),
            "layer_entropy_empirical_comparison_csv": str(
                paths.layer_entropy_empirical_comparison_csv
            ),
            "acf_empirical_comparison_csv": str(paths.acf_empirical_comparison_csv),
            "component_acf_empirical_comparison_csv": str(
                paths.component_acf_empirical_comparison_csv
            ),
            "abs_component_correlation_empirical_comparison_csv": str(
                paths.abs_component_correlation_empirical_comparison_csv
            ),
        },
        "rows": {
            "layer_volatility_simulations": int(len(volatility)),
            "layer_volatility_summary": int(len(volatility_summary)),
            "layer_entropy_simulations": int(len(entropy)),
            "layer_entropy_summary": int(len(entropy_summary)),
            "acf_simulations": int(len(acf)),
            "acf_summary": int(len(acf_summary)),
            "component_acf_simulations": int(len(component_acf)),
            "component_acf_summary": int(len(component_acf_summary)),
            "abs_component_correlation_simulations": int(len(corr)),
            "abs_component_correlation_summary": int(len(corr_summary)),
            "layer_volatility_empirical_comparison": int(len(volatility_comparison)),
            "layer_entropy_empirical_comparison": int(len(entropy_comparison)),
            "acf_empirical_comparison": int(len(acf_comparison)),
            "component_acf_empirical_comparison": int(len(component_acf_comparison)),
            "abs_component_correlation_empirical_comparison": int(len(corr_comparison)),
        },
    }


def compute_monte_carlo_comparisons(
    paths: MonteCarloMetricPaths | None = None,
    k: int = DEFAULT_K,
    return_acf_max_lag: int = RETURN_ACF_MAX_LAG,
    abs_return_acf_max_lag: int = ABS_RETURN_ACF_MAX_LAG,
    short_component_acf_max_lag: int = SHORT_COMPONENT_ACF_MAX_LAG,
    long_component_acf_max_lag: int = LONG_COMPONENT_ACF_MAX_LAG,
) -> dict[str, Any]:
    paths = paths or MonteCarloMetricPaths()
    require_positive_k(k)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    volatility = pd.read_csv(paths.layer_volatility_simulations_csv)
    volatility_summary = pd.read_csv(paths.layer_volatility_summary_csv)
    entropy = pd.read_csv(paths.layer_entropy_simulations_csv)
    entropy_summary = pd.read_csv(paths.layer_entropy_summary_csv)
    acf = pd.read_csv(paths.acf_simulations_csv)
    acf_summary = pd.read_csv(paths.acf_summary_csv)
    corr = pd.read_csv(paths.abs_component_correlation_simulations_csv)
    corr_summary = pd.read_csv(paths.abs_component_correlation_summary_csv)

    volatility_comparison = compare_empirical_volatility(
        paths.empirical_volatility_csv,
        volatility,
        volatility_summary,
        k=k,
    )
    entropy_comparison = compare_empirical_entropy(
        paths.empirical_entropy_csv,
        entropy,
        entropy_summary,
        k=k,
    )
    acf_comparison = compare_empirical_acf(
        paths.final_returns_csv,
        acf,
        acf_summary,
        return_acf_max_lag=return_acf_max_lag,
        abs_return_acf_max_lag=abs_return_acf_max_lag,
    )
    corr_comparison = compare_empirical_abs_component_correlation(
        paths.final_decomposition_csv,
        corr,
        corr_summary,
        k=k,
    )

    write_csv(
        volatility_comparison,
        paths.layer_volatility_empirical_comparison_csv,
        index=False,
    )
    write_csv(
        entropy_comparison,
        paths.layer_entropy_empirical_comparison_csv,
        index=False,
    )
    write_csv(acf_comparison, paths.acf_empirical_comparison_csv, index=False)
    component_acf_comparison = pd.DataFrame()
    if paths.component_acf_simulations_csv.exists() and paths.component_acf_summary_csv.exists():
        component_acf = pd.read_csv(paths.component_acf_simulations_csv)
        component_acf_summary = pd.read_csv(paths.component_acf_summary_csv)
        component_acf_comparison = compare_empirical_component_acf(
            paths.final_decomposition_csv,
            component_acf,
            component_acf_summary,
            k=k,
            short_max_lag=short_component_acf_max_lag,
            long_max_lag=long_component_acf_max_lag,
        )
        write_csv(
            component_acf_comparison,
            paths.component_acf_empirical_comparison_csv,
            index=False,
        )
    write_csv(
        corr_comparison,
        paths.abs_component_correlation_empirical_comparison_csv,
        index=False,
    )

    return {
        "outputs": {
            "layer_volatility_empirical_comparison_csv": str(
                paths.layer_volatility_empirical_comparison_csv
            ),
            "layer_entropy_empirical_comparison_csv": str(
                paths.layer_entropy_empirical_comparison_csv
            ),
            "acf_empirical_comparison_csv": str(paths.acf_empirical_comparison_csv),
            "component_acf_empirical_comparison_csv": str(
                paths.component_acf_empirical_comparison_csv
            ),
            "abs_component_correlation_empirical_comparison_csv": str(
                paths.abs_component_correlation_empirical_comparison_csv
            ),
        },
        "rows": {
            "layer_volatility_empirical_comparison": int(len(volatility_comparison)),
            "layer_entropy_empirical_comparison": int(len(entropy_comparison)),
            "acf_empirical_comparison": int(len(acf_comparison)),
            "component_acf_empirical_comparison": int(len(component_acf_comparison)),
            "abs_component_correlation_empirical_comparison": int(len(corr_comparison)),
        },
    }


def compute_volatility_rows(
    frame: pd.DataFrame,
    baseline_type: str,
    simulation_id: int,
    k: int,
    specs: list[ComponentSpec] | None = None,
) -> list[dict[str, Any]]:
    annualization_periods = 252 * 24 * (60 // BASE_INTERVAL_MINUTES)
    annualization_factor = float(np.sqrt(annualization_periods))
    specs = specs or component_specs(
        k,
        include_original=False,
        base_interval_minutes=BASE_INTERVAL_MINUTES,
    )
    n = len(frame)
    detail_energies: dict[str, float] = {}
    component_energies: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for spec in specs:
        values = frame[spec.name].astype(float).to_numpy()
        energy = float(np.dot(values, values))
        component_energies[spec.name] = energy
        if spec.kind == "detail":
            detail_energies[spec.name] = energy
        rows.append(
            {
                "baseline_type": baseline_type,
                "simulation_id": simulation_id,
                COMPONENT: spec.name,
                K: spec.scale,
                COMPONENT_TYPE: spec.kind,
                SCALE_MINUTES: spec.scale_minutes,
                SCALE_DAYS: spec.scale_days,
                ENERGY: energy,
                RMS_VOLATILITY: float(np.sqrt(energy / n)),
                ANNUALIZED_RMS_VOLATILITY: float(np.sqrt(energy / n) * annualization_factor),
                DETAIL_ENERGY_SHARE: np.nan,
                TOTAL_COMPONENT_ENERGY_SHARE: np.nan,
            }
        )

    detail_sum = float(sum(detail_energies.values()))
    total_sum = float(sum(component_energies.values()))
    for row in rows:
        component = row[COMPONENT]
        if row[COMPONENT_TYPE] == "detail":
            row[DETAIL_ENERGY_SHARE] = component_energies[component] / detail_sum
        row[TOTAL_COMPONENT_ENERGY_SHARE] = component_energies[component] / total_sum
    return rows


def compute_entropy_rows(
    frame: pd.DataFrame,
    baseline_type: str,
    simulation_id: int,
    k: int,
    embedding_dimension: int,
    delay: int,
    jitter_seed: int,
    jitter_magnitude: float,
    specs: list[ComponentSpec] | None = None,
) -> list[dict[str, Any]]:
    specs = specs or component_specs(
        k,
        include_original=False,
        base_interval_minutes=BASE_INTERVAL_MINUTES,
    )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        values = frame[spec.name].astype(float).to_numpy()
        compressed = compress_component(values, spec.name)
        component_seed = _component_jitter_seed(
            jitter_seed,
            f"{baseline_type}_{simulation_id:03d}",
            spec.name,
        )
        jittered = _add_jitter(compressed, component_seed, jitter_magnitude)
        entropy_result = _permutation_entropy(jittered, embedding_dimension, delay)
        rows.append(
            {
                "baseline_type": baseline_type,
                "simulation_id": simulation_id,
                COMPONENT: spec.name,
                K: spec.scale,
                COMPONENT_TYPE: spec.kind,
                SCALE_MINUTES: spec.scale_minutes,
                SCALE_DAYS: spec.scale_days,
                REPEAT_LENGTH: spec.repeat_length,
                EFFECTIVE_N: len(compressed),
                ORDINAL_WINDOWS: entropy_result[ORDINAL_WINDOWS],
                PERMUTATION_ENTROPY: entropy_result[PERMUTATION_ENTROPY],
                NORMALIZED_ENTROPY: entropy_result[NORMALIZED_ENTROPY],
            }
        )
    return rows


def compute_acf_rows(
    values: np.ndarray,
    baseline_type: str,
    simulation_id: int,
    return_acf_max_lag: int,
    abs_return_acf_max_lag: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for acf_kind, acf_values, max_lag in [
        ("return", autocorrelation(values, return_acf_max_lag), return_acf_max_lag),
        (
            "absolute_return",
            autocorrelation(np.abs(values), abs_return_acf_max_lag),
            abs_return_acf_max_lag,
        ),
    ]:
        rows.extend(
            {
                "baseline_type": baseline_type,
                "simulation_id": simulation_id,
                "acf_kind": acf_kind,
                "lag": lag,
                "max_lag": max_lag,
                "acf": float(value),
            }
            for lag, value in enumerate(acf_values, start=1)
        )
    return rows


def compute_component_acf_rows(
    frame: pd.DataFrame,
    baseline_type: str,
    simulation_id: int,
    k: int,
    short_max_lag: int,
    long_max_lag: int,
    specs: list[ComponentSpec] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = specs or component_specs(
        k,
        include_original=False,
        base_interval_minutes=BASE_INTERVAL_MINUTES,
    )
    for spec in specs:
        max_lag = short_max_lag if spec.kind == "detail" and spec.scale <= 6 else long_max_lag
        values = frame[spec.name].astype(float).to_numpy()
        for acf_kind, series_values in [
            ("component", values),
            ("absolute_component", np.abs(values)),
        ]:
            lags, acf_values = compressed_layer_autocorrelation(
                series_values,
                spec.name,
                max_lag,
            )
            rows.extend(
                {
                    "baseline_type": baseline_type,
                    "simulation_id": simulation_id,
                    COMPONENT: spec.name,
                    K: spec.scale,
                    COMPONENT_TYPE: spec.kind,
                    SCALE_MINUTES: spec.scale_minutes,
                    SCALE_DAYS: spec.scale_days,
                    "acf_kind": acf_kind,
                    "lag": int(lag),
                    "compressed_lag": int(compressed_lag),
                    "max_lag": int(max_lag),
                    "acf": float(value),
                }
                for compressed_lag, (lag, value) in enumerate(
                    zip(lags, acf_values, strict=True),
                    start=1,
                )
            )
    return rows


def compute_abs_component_correlation_rows(
    frame: pd.DataFrame,
    baseline_type: str,
    simulation_id: int,
    k: int,
    components: list[str] | None = None,
) -> list[dict[str, Any]]:
    components = components or decomposition_components(k, include_original=False)
    corr = absolute_component_correlation(frame, components)
    return [
        {
            "baseline_type": baseline_type,
            "simulation_id": simulation_id,
            "component_i": component_i,
            "component_j": component_j,
            "correlation_abs": float(corr.loc[component_i, component_j]),
        }
        for component_i, component_j in itertools.product(components, components)
    ]


def summarize_metrics(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_cols, group_key, strict=True))
        for metric in value_cols:
            values = group[metric].dropna().astype(float).to_numpy()
            if len(values) == 0:
                continue
            p05, median, p95 = np.quantile(
                values,
                SUMMARY_QUANTILES,
                method=MONTE_CARLO_BASELINE_QUANTILE_METHOD,
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_simulations": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(median),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "p05": float(p05),
                    "p95": float(p95),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "quantile_method": MONTE_CARLO_BASELINE_QUANTILE_METHOD,
                }
            )
    return pd.DataFrame(rows)


def compare_empirical_volatility(
    empirical_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    empirical = pd.read_csv(empirical_csv)
    empirical = empirical[empirical["series"] == SERIES_FINAL].copy()
    value_cols = [
        ENERGY,
        RMS_VOLATILITY,
        ANNUALIZED_RMS_VOLATILITY,
        DETAIL_ENERGY_SHARE,
        TOTAL_COMPONENT_ENERGY_SHARE,
    ]
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=[
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
        ],
        value_cols=value_cols,
    )


def compare_empirical_entropy(
    empirical_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    empirical = pd.read_csv(empirical_csv)
    empirical = empirical[empirical["series"] == SERIES_FINAL].copy()
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=[
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
            REPEAT_LENGTH,
        ],
        value_cols=[
            EFFECTIVE_N,
            ORDINAL_WINDOWS,
            PERMUTATION_ENTROPY,
            NORMALIZED_ENTROPY,
        ],
    )


def compare_empirical_acf(
    final_returns_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    return_acf_max_lag: int,
    abs_return_acf_max_lag: int,
) -> pd.DataFrame:
    returns = pd.read_csv(final_returns_csv, usecols=[LOG_RETURN])
    values = returns[LOG_RETURN].astype(float).to_numpy()
    empirical = pd.DataFrame(
        [
            *(
                {
                    "acf_kind": "return",
                    "lag": lag,
                    "max_lag": return_acf_max_lag,
                    "acf": float(value),
                }
                for lag, value in enumerate(
                    autocorrelation(values, return_acf_max_lag),
                    start=1,
                )
            ),
            *(
                {
                    "acf_kind": "absolute_return",
                    "lag": lag,
                    "max_lag": abs_return_acf_max_lag,
                    "acf": float(value),
                }
                for lag, value in enumerate(
                    autocorrelation(np.abs(values), abs_return_acf_max_lag),
                    start=1,
                )
            ),
        ]
    )
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=["acf_kind", "lag", "max_lag"],
        value_cols=["acf"],
    )


def compare_empirical_component_acf(
    final_decomposition_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
    short_max_lag: int,
    long_max_lag: int,
) -> pd.DataFrame:
    components = decomposition_components(k, include_original=False)
    frame = pd.read_csv(final_decomposition_csv, usecols=components)
    empirical = pd.DataFrame(
        compute_component_acf_rows(
            frame,
            baseline_type=SERIES_FINAL,
            simulation_id=0,
            k=k,
            short_max_lag=short_max_lag,
            long_max_lag=long_max_lag,
        )
    )
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=[
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
            "acf_kind",
            "lag",
            "compressed_lag",
            "max_lag",
        ],
        value_cols=["acf"],
    )


def compare_empirical_abs_component_correlation(
    final_decomposition_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    components = decomposition_components(k, include_original=False)
    frame = pd.read_csv(final_decomposition_csv, usecols=components)
    corr = absolute_component_correlation(frame, components)
    empirical = pd.DataFrame(
        [
            {
                "component_i": component_i,
                "component_j": component_j,
                "correlation_abs": float(corr.loc[component_i, component_j]),
            }
            for component_i, component_j in itertools.product(components, components)
        ]
    )
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=["component_i", "component_j"],
        value_cols=["correlation_abs"],
    )


def compare_empirical_metric_table(
    empirical: pd.DataFrame,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    index_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    empirical = normalize_comparison_keys(empirical, index_cols)
    simulations = normalize_comparison_keys(simulations, index_cols)
    summary = normalize_comparison_keys(summary, index_cols)
    empirical_long = empirical.melt(
        id_vars=index_cols,
        value_vars=value_cols,
        var_name="metric",
        value_name="empirical_value",
    ).dropna(subset=["empirical_value"])
    simulations_long = simulations.melt(
        id_vars=["baseline_type", "simulation_id", *index_cols],
        value_vars=value_cols,
        var_name="metric",
        value_name="simulated_value",
    ).dropna(subset=["simulated_value"])

    keys = ["baseline_type", *index_cols, "metric"]
    comparison = summary.merge(
        empirical_long,
        on=[*index_cols, "metric"],
        how="inner",
    )
    joined = simulations_long.merge(
        comparison[[*keys, "empirical_value"]],
        on=keys,
        how="inner",
    )
    ranks = (
        joined.assign(le_empirical=joined["simulated_value"] <= joined["empirical_value"])
        .groupby(keys, dropna=False)
        .agg(
            percentile_rank=("le_empirical", "mean"),
            n_simulations=("simulated_value", "count"),
        )
        .reset_index()
    )

    output = comparison.merge(ranks, on=keys, how="inner")
    if "n_simulations_y" in output.columns:
        output["n_simulations"] = output["n_simulations_y"]
        output = output.drop(
            columns=[
                column
                for column in ["n_simulations_x", "n_simulations_y"]
                if column in output.columns
            ]
        )
    output = output.rename(
        columns={
            "median": "baseline_median",
            "p05": "baseline_p05",
            "p95": "baseline_p95",
        }
    )
    output["difference_from_median"] = (
        output["empirical_value"] - output["baseline_median"]
    )
    output["inside_envelope"] = (
        (output["baseline_p05"] <= output["empirical_value"])
        & (output["empirical_value"] <= output["baseline_p95"])
    )
    output["above_envelope"] = output["empirical_value"] > output["baseline_p95"]
    output["below_envelope"] = output["empirical_value"] < output["baseline_p05"]
    output["quantile_method"] = MONTE_CARLO_BASELINE_QUANTILE_METHOD

    columns = [
        "baseline_type",
        *index_cols,
        "metric",
        "empirical_value",
        "baseline_median",
        "baseline_p05",
        "baseline_p95",
        "difference_from_median",
        "percentile_rank",
        "inside_envelope",
        "above_envelope",
        "below_envelope",
        "n_simulations",
        "quantile_method",
    ]
    return output[columns].sort_values(["baseline_type", *index_cols, "metric"])


def normalize_comparison_keys(frame: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in index_cols:
        if column in output.columns and pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].round(12)
    return output

