"""Metric tables and baseline envelopes for Monte Carlo baseline simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from multi_scale_volatility.app.parallel import effective_worker_count, process_pool_map
from multi_scale_volatility.app.runtime import RuntimeTracker, get_logger, start_timer
from multi_scale_volatility.core.components import component_specs
from multi_scale_volatility.core.config.names import (
    ANNUALIZED_RMS_VOLATILITY,
    BASE_INTERVAL_MINUTES,
    COMPONENT,
    COMPONENT_TYPE,
    DEFAULT_K,
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
from multi_scale_volatility.core.config.paths import (
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    LAYER_ENTROPY_CSV,
    MC_ABS_COMPONENT_CORRELATION_EMPIRICAL_COMPARISON_CSV,
    MC_ABS_COMPONENT_CORRELATION_SIMULATIONS_CSV,
    MC_ABS_COMPONENT_CORRELATION_SUMMARY_CSV,
    MC_ACF_EMPIRICAL_COMPARISON_CSV,
    MC_ACF_SIMULATIONS_CSV,
    MC_ACF_SUMMARY_CSV,
    MC_COMPONENT_ACF_EMPIRICAL_COMPARISON_CSV,
    MC_COMPONENT_ACF_SIMULATIONS_CSV,
    MC_COMPONENT_ACF_SUMMARY_CSV,
    MC_LAYER_ENTROPY_EMPIRICAL_COMPARISON_CSV,
    MC_LAYER_ENTROPY_SIMULATIONS_CSV,
    MC_LAYER_ENTROPY_SUMMARY_CSV,
    MC_LAYER_VOLATILITY_EMPIRICAL_COMPARISON_CSV,
    MC_LAYER_VOLATILITY_SIMULATIONS_CSV,
    MC_LAYER_VOLATILITY_SUMMARY_CSV,
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV,
    MONTE_CARLO_BASELINES_RESULTS_DIR,
    VOLATILITY_CSV,
)
from multi_scale_volatility.core.io import write_csv
from multi_scale_volatility.core.utils.validation import require_positive_k
from multi_scale_volatility.research.global_diagnosis.entropy import (
    DELAY,
    EMBEDDING_DIMENSION,
    JITTER_MAGNITUDE,
    JITTER_SEED,
)
from multi_scale_volatility.research.global_diagnosis.monte_carlo_rows import (
    _compute_simulation_metrics_worker,
)
from multi_scale_volatility.research.global_diagnosis.monte_carlo_summaries import (
    compare_empirical_abs_component_correlation,
    compare_empirical_acf,
    compare_empirical_component_acf,
    compare_empirical_entropy,
    compare_empirical_volatility,
    summarize_metrics,
)

RETURN_ACF_MAX_LAG = 288
ABS_RETURN_ACF_MAX_LAG = 1440
SHORT_COMPONENT_ACF_MAX_LAG = 1440
LONG_COMPONENT_ACF_MAX_LAG = 6336
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

    effective_max_workers = effective_worker_count(max_workers)
    logger.info(
        "Computing Monte Carlo metrics for %s simulations with %s workers",
        len(worker_args),
        effective_max_workers,
    )
    for index, result in process_pool_map(
        _compute_simulation_metrics_worker,
        worker_args,
        max_workers=max_workers,
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


