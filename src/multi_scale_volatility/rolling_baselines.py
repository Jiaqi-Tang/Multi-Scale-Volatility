"""Rolling correlation envelopes for Monte Carlo baseline simulations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.config.names import (
    COMPONENT,
    DETAIL_ENERGY_SHARE,
    LOG_RETURN,
    RMS_VOLATILITY,
)
from multi_scale_volatility.config.paths import (
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    ROLLING_BASELINE_CORRELATION_EMPIRICAL_COMPARISON_CSV,
    ROLLING_BASELINE_CORRELATION_SIMULATIONS_CSV,
    ROLLING_BASELINE_CORRELATION_SUMMARY_CSV,
    ROLLING_BASELINE_REPORT_JSON,
    ROLLING_BASELINE_RESULTS_DIR,
    ROLLING_BASELINE_RUNTIME_LOG_CSV,
    ROLLING_LAYER_VOLATILITY_CSV,
)
from multi_scale_volatility.decomposition import decompose_values
from multi_scale_volatility.io import write_csv, write_json
from multi_scale_volatility.monte_carlo_metrics import (
    compare_empirical_metric_table,
    summarize_metrics,
)
from multi_scale_volatility.rolling import (
    ROLLING_K,
    ROLLING_STEP_SIZE,
    ROLLING_WINDOW_LENGTHS,
    rolling_window_specs,
)
from multi_scale_volatility.runtime import (
    RuntimeTracker,
    get_logger,
    runtime_row,
    start_timer,
)
from multi_scale_volatility.utils.validation import require_finite_array, require_positive_k

ROLLING_CORRELATION_METRICS = (
    "rms_volatility_correlation",
    "detail_energy_share_percentile_correlation",
)
SUMMARY_QUANTILES = (0.05, 0.5, 0.95)
RUNTIME_LOG_BATCH_SIZE = 10
logger = get_logger(__name__)


@dataclass(frozen=True)
class RollingBaselineCorrelationPaths:
    audit_csv: Path = MONTE_CARLO_BASELINE_AUDIT_CSV
    empirical_layer_volatility_csv: Path = ROLLING_LAYER_VOLATILITY_CSV
    output_dir: Path = ROLLING_BASELINE_RESULTS_DIR
    runtime_log_csv: Path = ROLLING_BASELINE_RUNTIME_LOG_CSV

    @property
    def simulations_csv(self) -> Path:
        return self.output_dir / ROLLING_BASELINE_CORRELATION_SIMULATIONS_CSV.name

    @property
    def summary_csv(self) -> Path:
        return self.output_dir / ROLLING_BASELINE_CORRELATION_SUMMARY_CSV.name

    @property
    def empirical_comparison_csv(self) -> Path:
        return self.output_dir / ROLLING_BASELINE_CORRELATION_EMPIRICAL_COMPARISON_CSV.name

    @property
    def report_json(self) -> Path:
        return self.output_dir / ROLLING_BASELINE_REPORT_JSON.name


@dataclass(frozen=True)
class RollingBaselineWorkerResult:
    rows: list[dict[str, Any]]
    runtime_rows: list[dict[str, Any]]
    status: str
    error_message: str


def compute_rolling_baseline_correlations(
    paths: RollingBaselineCorrelationPaths | None = None,
    window_lengths: tuple[int, ...] = ROLLING_WINDOW_LENGTHS,
    step_size: int = ROLLING_STEP_SIZE,
    k: int = ROLLING_K,
    max_workers: int | None = None,
    max_simulations_per_type: int | None = None,
) -> dict[str, Any]:
    paths = paths or RollingBaselineCorrelationPaths()
    require_positive_k(k)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(paths.audit_csv)
    if audit.empty:
        raise ValueError(f"Audit table is empty: {paths.audit_csv}")
    if max_simulations_per_type is not None:
        audit = (
            audit.sort_values(["baseline_type", "simulation_id"])
            .groupby("baseline_type", group_keys=False)
            .head(max_simulations_per_type)
            .reset_index(drop=True)
        )

    worker_args = [
        (record, window_lengths, step_size, k)
        for record in audit.to_dict("records")
    ]
    effective_max_workers = max_workers or min(4, os.cpu_count() or 1)
    tracker = RuntimeTracker(paths.runtime_log_csv, flush_every=RUNTIME_LOG_BATCH_SIZE)
    stage_timer = start_timer()
    rows: list[dict[str, Any]] = []

    logger.info(
        "Computing rolling baseline correlations for %s simulations with %s workers",
        len(worker_args),
        effective_max_workers,
    )
    with ProcessPoolExecutor(max_workers=effective_max_workers) as executor:
        for index, result in enumerate(
            executor.map(_rolling_baseline_worker, worker_args),
            start=1,
        ):
            rows.extend(result.rows)
            tracker.extend(
                result.runtime_rows,
                flush=index % RUNTIME_LOG_BATCH_SIZE == 0 or result.status != "success",
            )
            if result.status != "success":
                raise RuntimeError(result.error_message)
            if index % RUNTIME_LOG_BATCH_SIZE == 0:
                logger.info(
                    "Computed rolling baseline correlations for %s/%s simulations",
                    index,
                    len(worker_args),
                )
    tracker.flush()

    simulations = pd.DataFrame(rows)
    write_csv(simulations, paths.simulations_csv, index=False)

    summary = summarize_metrics(
        simulations,
        group_cols=[
            "baseline_type",
            "window_length",
            "correlation_kind",
            "component_i",
            "component_j",
        ],
        value_cols=["correlation"],
    )
    write_csv(summary, paths.summary_csv, index=False)

    empirical = compute_empirical_rolling_correlation_rows(
        paths.empirical_layer_volatility_csv,
        window_lengths=window_lengths,
    )
    comparison = compare_empirical_metric_table(
        pd.DataFrame(empirical),
        simulations,
        summary,
        index_cols=[
            "window_length",
            "correlation_kind",
            "component_i",
            "component_j",
        ],
        value_cols=["correlation"],
    )
    comparison["outside_envelope"] = ~comparison["inside_envelope"].astype(bool)
    write_csv(comparison, paths.empirical_comparison_csv, index=False)

    tracker.record(
        stage="rolling_baselines",
        operation="compute_rolling_baseline_correlations",
        started_at_utc=stage_timer.started_at_utc,
        elapsed_seconds=stage_timer.elapsed_seconds,
        rows_in=len(audit),
        rows_out=len(simulations),
        output_path=(
            f"{paths.simulations_csv};{paths.summary_csv};"
            f"{paths.empirical_comparison_csv}"
        ),
        flush=True,
    )

    report = {
        "audit_csv": str(paths.audit_csv),
        "empirical_layer_volatility_csv": str(paths.empirical_layer_volatility_csv),
        "simulations_csv": str(paths.simulations_csv),
        "summary_csv": str(paths.summary_csv),
        "empirical_comparison_csv": str(paths.empirical_comparison_csv),
        "runtime_log_csv": str(paths.runtime_log_csv),
        "window_lengths": list(window_lengths),
        "step_size": int(step_size),
        "K_roll": int(k),
        "baseline_simulations": audit.groupby("baseline_type").size().to_dict(),
        "correlation_kinds": list(ROLLING_CORRELATION_METRICS),
        "rows": {
            "simulations": int(len(simulations)),
            "summary": int(len(summary)),
            "empirical_comparison": int(len(comparison)),
        },
    }
    write_json(paths.report_json, report)
    return report


def _rolling_baseline_worker(args: tuple[Any, ...]) -> RollingBaselineWorkerResult:
    record, window_lengths, step_size, k = args
    baseline_type = str(record["baseline_type"])
    simulation_id = int(record["simulation_id"])
    return_parquet = Path(record["return_parquet"])
    n = int(record["n"])
    timer = start_timer()
    runtime_rows: list[dict[str, Any]] = []
    try:
        returns = pd.read_parquet(return_parquet, columns=[LOG_RETURN])
        values = returns[LOG_RETURN].astype(float).to_numpy()
        require_finite_array(values, f"{baseline_type} {simulation_id} returns")
        rows = compute_rolling_correlation_rows_for_values(
            values,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            window_lengths=window_lengths,
            step_size=step_size,
            k=k,
        )
        status = "success"
        error_message = ""
    except Exception as error:
        rows = []
        status = "error"
        error_message = str(error)

    runtime_rows.append(
        runtime_row(
            run_id="",
            stage="rolling_baselines",
            operation="compute_simulation_rolling_correlations",
            started_at_utc=timer.started_at_utc,
            elapsed_seconds=timer.elapsed_seconds,
            baseline_type=baseline_type,
            simulation_id=simulation_id,
            status=status,
            rows_in=n,
            rows_out=len(rows),
            output_path=str(return_parquet),
            error_message=error_message,
        )
    )
    return RollingBaselineWorkerResult(
        rows=rows,
        runtime_rows=runtime_rows,
        status=status,
        error_message=error_message,
    )


def compute_rolling_correlation_rows_for_values(
    values: np.ndarray,
    baseline_type: str,
    simulation_id: int,
    window_lengths: tuple[int, ...],
    step_size: int,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    components = detail_components(k)
    for window_length in window_lengths:
        rms_matrix, share_matrix = rolling_metric_matrices(
            values,
            window_length=window_length,
            step_size=step_size,
            k=k,
        )
        rows.extend(
            matrix_correlation_rows(
                rms_matrix,
                components=components,
                window_length=window_length,
                correlation_kind="rms_volatility_correlation",
                baseline_type=baseline_type,
                simulation_id=simulation_id,
            )
        )
        share_percentiles = percentile_columns(share_matrix)
        rows.extend(
            matrix_correlation_rows(
                share_percentiles,
                components=components,
                window_length=window_length,
                correlation_kind="detail_energy_share_percentile_correlation",
                baseline_type=baseline_type,
                simulation_id=simulation_id,
            )
        )
    return rows


def rolling_metric_matrices(
    values: np.ndarray,
    window_length: int,
    step_size: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    specs = rolling_window_specs(len(values), window_length, step_size)
    rms_matrix = np.empty((len(specs), k), dtype=float)
    share_matrix = np.empty((len(specs), k), dtype=float)
    for row_index, spec in enumerate(specs):
        window_values = values[spec.start_index : spec.end_index + 1]
        details, _approximation = decompose_values(window_values, k=k)
        energies = np.array([np.dot(detail, detail) for detail in details], dtype=float)
        detail_energy_sum = float(energies.sum())
        if detail_energy_sum <= 0:
            raise ValueError(
                f"Detail energy sum is non-positive for W={window_length}, "
                f"window_id={spec.window_id}"
            )
        rms_matrix[row_index, :] = np.sqrt(energies / window_length)
        share_matrix[row_index, :] = energies / detail_energy_sum
    return rms_matrix, share_matrix


def percentile_columns(values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    ranks = frame.rank(method="average", axis=0) - 1.0
    denominator = len(frame) - 1.0
    if denominator <= 0:
        return np.full_like(values, np.nan, dtype=float)
    return (ranks / denominator).to_numpy(dtype=float)


def matrix_correlation_rows(
    matrix: np.ndarray,
    components: list[str],
    window_length: int,
    correlation_kind: str,
    baseline_type: str,
    simulation_id: int,
) -> list[dict[str, Any]]:
    correlation = np.corrcoef(matrix, rowvar=False)
    return [
        {
            "baseline_type": baseline_type,
            "simulation_id": simulation_id,
            "window_length": int(window_length),
            "correlation_kind": correlation_kind,
            "component_i": component_i,
            "component_j": component_j,
            "correlation": float(correlation[row_i, col_i]),
        }
        for row_i, component_i in enumerate(components)
        for col_i, component_j in enumerate(components)
    ]


def compute_empirical_rolling_correlation_rows(
    layer_volatility_csv: Path,
    window_lengths: tuple[int, ...] = ROLLING_WINDOW_LENGTHS,
    k: int = ROLLING_K,
) -> list[dict[str, Any]]:
    layer = pd.read_csv(layer_volatility_csv)
    rows: list[dict[str, Any]] = []
    components = detail_components(k)
    for window_length in window_lengths:
        subset = layer[
            (layer["window_length"] == window_length)
            & (layer[COMPONENT].isin(components))
        ].copy()
        rms = layer_metric_matrix(subset, RMS_VOLATILITY, components)
        rows.extend(
            empirical_matrix_correlation_rows(
                rms,
                components,
                window_length,
                "rms_volatility_correlation",
            )
        )
        shares = layer_metric_matrix(subset, DETAIL_ENERGY_SHARE, components)
        share_percentiles = percentile_columns(shares)
        rows.extend(
            empirical_matrix_correlation_rows(
                share_percentiles,
                components,
                window_length,
                "detail_energy_share_percentile_correlation",
            )
        )
    return rows


def layer_metric_matrix(
    frame: pd.DataFrame,
    metric: str,
    components: list[str],
) -> np.ndarray:
    matrix = (
        frame.pivot(index="window_id", columns=COMPONENT, values=metric)
        .reindex(columns=components)
        .sort_index()
    )
    return matrix.to_numpy(dtype=float)


def empirical_matrix_correlation_rows(
    matrix: np.ndarray,
    components: list[str],
    window_length: int,
    correlation_kind: str,
) -> list[dict[str, Any]]:
    correlation = np.corrcoef(matrix, rowvar=False)
    return [
        {
            "window_length": int(window_length),
            "correlation_kind": correlation_kind,
            "component_i": component_i,
            "component_j": component_j,
            "correlation": float(correlation[row_i, col_i]),
        }
        for row_i, component_i in enumerate(components)
        for col_i, component_j in enumerate(components)
    ]


def detail_components(k: int) -> list[str]:
    return [f"D_{scale:02d}" for scale in range(1, k + 1)]
