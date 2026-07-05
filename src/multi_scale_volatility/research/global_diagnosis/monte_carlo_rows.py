"""Row builders for Monte Carlo baseline metric tables."""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.app.runtime import runtime_row, start_timer
from multi_scale_volatility.core.components import ComponentSpec, component_specs, compress_component
from multi_scale_volatility.core.config.names import (
    ANNUALIZED_RMS_VOLATILITY,
    BASE_INTERVAL_MINUTES,
    COMPONENT,
    COMPONENT_TYPE,
    DETAIL_ENERGY_SHARE,
    EFFECTIVE_N,
    ENERGY,
    K,
    LOG_RETURN,
    NORMALIZED_ENTROPY,
    ORDINAL_WINDOWS,
    PERMUTATION_ENTROPY,
    REPEAT_LENGTH,
    RMS_VOLATILITY,
    SCALE_DAYS,
    SCALE_MINUTES,
    TOTAL_COMPONENT_ENERGY_SHARE,
)
from multi_scale_volatility.core.stats import (
    absolute_component_correlation,
    autocorrelation,
    compressed_layer_autocorrelation,
)
from multi_scale_volatility.core.utils.validation import require_finite_array, require_positive_k
from multi_scale_volatility.research.global_diagnosis.entropy import (
    _add_jitter,
    _component_jitter_seed,
    _permutation_entropy,
)

RETURN_ACF_MAX_LAG = 288
ABS_RETURN_ACF_MAX_LAG = 1440
SHORT_COMPONENT_ACF_MAX_LAG = 1440
LONG_COMPONENT_ACF_MAX_LAG = 6336

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


