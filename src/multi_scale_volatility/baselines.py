"""Create Monte Carlo baseline return and decomposition artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.config.names import INDEX, LOG_RETURN, ORIGINAL, TIMESTAMP_UTC
from multi_scale_volatility.config.names import (
    BASE_INTERVAL_MINUTES,
    DEFAULT_K,
    MONTE_CARLO_BASELINE_MASTER_SEED,
    MONTE_CARLO_BASELINE_QUANTILE_METHOD,
    MONTE_CARLO_BASELINE_QUANTILES,
    MONTE_CARLO_BASELINE_SIMULATIONS,
    MONTE_CARLO_BASELINE_TYPES,
)
from multi_scale_volatility.config.paths import (
    FINAL_RETURNS_CSV,
    MONTE_CARLO_BASELINES_DATA_DIR,
    MONTE_CARLO_BASELINES_RESULTS_DIR,
)
from multi_scale_volatility.decomposition import RECONSTRUCTION_TOLERANCE, decompose_values
from multi_scale_volatility.io import write_csv, write_parquet
from multi_scale_volatility.io import write_json
from multi_scale_volatility.runtime import RuntimeTracker, get_logger, start_timer
from multi_scale_volatility.utils.validation import require_finite_array, require_positive_k

RUNTIME_LOG_BATCH_SIZE = 10
logger = get_logger(__name__)


@dataclass(frozen=True)
class BaselinePaths:
    input_csv: Path = FINAL_RETURNS_CSV
    data_dir: Path = MONTE_CARLO_BASELINES_DATA_DIR
    results_dir: Path = MONTE_CARLO_BASELINES_RESULTS_DIR

    @property
    def returns_dir(self) -> Path:
        return self.data_dir / "returns"

    @property
    def decomposition_dir(self) -> Path:
        return self.data_dir / "decomposition"

    @property
    def config_json(self) -> Path:
        return self.results_dir / "monte_carlo_config.json"

    @property
    def audit_csv(self) -> Path:
        return self.results_dir / "baseline_simulation_audit.csv"

    @property
    def runtime_log_csv(self) -> Path:
        return self.results_dir / "runtime_log.csv"


def create_baselines(
    paths: BaselinePaths | None = None,
    k: int = DEFAULT_K,
    n_simulations: int = MONTE_CARLO_BASELINE_SIMULATIONS,
    master_seed: int = MONTE_CARLO_BASELINE_MASTER_SEED,
) -> dict[str, Any]:
    paths = paths or BaselinePaths()
    require_positive_k(k)

    data = pd.read_csv(paths.input_csv, usecols=[TIMESTAMP_UTC, LOG_RETURN])
    if data.empty:
        raise ValueError(f"Input dataset is empty: {paths.input_csv}")

    timestamps = data[TIMESTAMP_UTC].copy()
    returns = data[LOG_RETURN].astype(float).to_numpy()
    require_finite_array(returns, f"Input {LOG_RETURN} values in {paths.input_csv}")

    n = len(returns)
    block_size_max = 2**k
    if n % block_size_max != 0:
        raise ValueError(
            f"Input length {n} is not divisible by 2**{k} ({block_size_max}): "
            f"{paths.input_csv}"
        )

    empirical_mean = float(np.mean(returns))
    empirical_variance = float(np.var(returns, ddof=0))
    empirical_std = float(np.sqrt(empirical_variance))

    audit_rows: list[dict[str, Any]] = []
    tracker = RuntimeTracker(paths.runtime_log_csv, flush_every=RUNTIME_LOG_BATCH_SIZE)
    stage_timer = start_timer()
    total_simulations = len(MONTE_CARLO_BASELINE_TYPES) * n_simulations
    completed_simulations = 0
    logger.info("Generating %s Monte Carlo baseline simulations", total_simulations)
    for baseline_type in MONTE_CARLO_BASELINE_TYPES:
        for simulation_id in range(n_simulations):
            operation_timer = start_timer()
            return_path = baseline_return_path(
                paths.returns_dir,
                baseline_type,
                simulation_id,
            )
            decomposition_path = baseline_decomposition_path(
                paths.decomposition_dir,
                baseline_type,
                simulation_id,
            )
            status = "success"
            error_message = ""
            seed = derive_seed(master_seed, baseline_type, simulation_id)
            try:
                rng = np.random.default_rng(seed)
                if baseline_type == "shuffle":
                    baseline_returns = rng.permutation(returns)
                elif baseline_type == "gaussian":
                    baseline_returns = rng.normal(
                        loc=0.0,
                        scale=empirical_std,
                        size=n,
                    )
                else:
                    raise ValueError(f"Unsupported baseline type: {baseline_type}")

                write_parquet(
                    pd.DataFrame(
                        {
                            TIMESTAMP_UTC: timestamps,
                            LOG_RETURN: baseline_returns,
                        }
                    ),
                    return_path,
                    index=False,
                )
                max_abs_error = write_baseline_decomposition(
                    baseline_returns,
                    timestamps,
                    decomposition_path,
                    k=k,
                )
                audit_rows.append(
                    {
                        "baseline_type": baseline_type,
                        "simulation_id": simulation_id,
                        "seed": seed,
                        "return_parquet": str(return_path),
                        "decomposition_parquet": str(decomposition_path),
                        "n": int(n),
                        "mean": float(np.mean(baseline_returns)),
                        "population_variance": float(np.var(baseline_returns, ddof=0)),
                        "population_std": float(np.std(baseline_returns, ddof=0)),
                        "min": float(np.min(baseline_returns)),
                        "max": float(np.max(baseline_returns)),
                        "max_abs_reconstruction_error": max_abs_error,
                    }
                )
            except Exception as error:
                status = "error"
                error_message = str(error)
                raise
            finally:
                tracker.record(
                    stage="baselines",
                    operation="generate_write_simulation",
                    started_at_utc=operation_timer.started_at_utc,
                    elapsed_seconds=operation_timer.elapsed_seconds,
                    baseline_type=baseline_type,
                    simulation_id=simulation_id,
                    status=status,
                    rows_in=n,
                    rows_out=n if status == "success" else 0,
                    output_path=f"{return_path};{decomposition_path}",
                    error_message=error_message,
                    flush=status != "success",
                )
                completed_simulations += 1
                if completed_simulations % RUNTIME_LOG_BATCH_SIZE == 0:
                    logger.info(
                        "Generated %s/%s baseline simulations",
                        completed_simulations,
                        total_simulations,
                    )

    tracker.record(
        stage="baselines",
        operation="create_baselines",
        started_at_utc=stage_timer.started_at_utc,
        elapsed_seconds=stage_timer.elapsed_seconds,
        rows_in=n,
        rows_out=len(audit_rows),
        output_path=f"{paths.audit_csv};{paths.config_json}",
        flush=True,
    )

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_rows)
    write_csv(audit, paths.audit_csv, index=False)

    report = {
        "input_csv": str(paths.input_csv),
        "input_rows": int(n),
        "timestamp_start_utc": str(timestamps.iloc[0]),
        "timestamp_end_utc": str(timestamps.iloc[-1]),
        "K": int(k),
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "block_size_max": int(block_size_max),
        "max_scale_minutes": int(BASE_INTERVAL_MINUTES * block_size_max),
        "max_scale_days": BASE_INTERVAL_MINUTES * block_size_max / (60 * 24),
        "n_simulations": int(n_simulations),
        "baseline_types": list(MONTE_CARLO_BASELINE_TYPES),
        "master_seed": int(master_seed),
        "quantiles": list(MONTE_CARLO_BASELINE_QUANTILES),
        "quantile_method": MONTE_CARLO_BASELINE_QUANTILE_METHOD,
        "returns_dir": str(paths.returns_dir),
        "decomposition_dir": str(paths.decomposition_dir),
        "audit_csv": str(paths.audit_csv),
        "runtime_log_csv": str(paths.runtime_log_csv),
        "empirical_mean_log_return": empirical_mean,
        "empirical_population_variance_log_return": empirical_variance,
        "empirical_population_std_log_return": empirical_std,
        "gaussian_sampling": {
            "mean": 0.0,
            "variance_source": "empirical_population_variance",
            "target_population_variance": empirical_variance,
            "exact_rescaling": False,
        },
        "shuffle_sampling": {
            "rule": "random_permutation_of_empirical_returns",
        },
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "simulation_counts": audit.groupby("baseline_type").size().to_dict(),
    }
    write_json(paths.config_json, report)
    logger.info("Wrote baseline audit to %s", paths.audit_csv)
    logger.info("Wrote baseline config to %s", paths.config_json)
    return report


def derive_seed(master_seed: int, baseline_type: str, simulation_id: int) -> int:
    digest = hashlib.sha256(
        f"{master_seed}:{baseline_type}:{simulation_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def baseline_return_path(
    returns_dir: Path,
    baseline_type: str,
    simulation_id: int,
) -> Path:
    return returns_dir / baseline_type / f"{baseline_type}_sim_{simulation_id:03d}.parquet"


def baseline_decomposition_path(
    decomposition_dir: Path,
    baseline_type: str,
    simulation_id: int,
) -> Path:
    return (
        decomposition_dir
        / baseline_type
        / f"{baseline_type}_decomposition_sim_{simulation_id:03d}.parquet"
    )


def write_baseline_decomposition(
    values: np.ndarray,
    timestamps: pd.Series,
    output_path: Path,
    k: int,
) -> float:
    details, final_approximation = decompose_values(values, k=k)
    reconstruction = final_approximation.copy()
    for detail in details:
        reconstruction += detail
    error = values - reconstruction
    max_abs_error = float(np.max(np.abs(error)))
    if max_abs_error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"Reconstruction error {max_abs_error} exceeds tolerance "
            f"{RECONSTRUCTION_TOLERANCE} for {output_path}"
        )

    output = pd.DataFrame(
        {
            INDEX: np.arange(len(values), dtype=np.int64),
            TIMESTAMP_UTC: timestamps,
            ORIGINAL: values,
        }
    )
    for scale, detail in enumerate(details, start=1):
        output[f"D_{scale:02d}"] = detail
    output[f"A_{k:02d}"] = final_approximation
    write_parquet(output, output_path, index=False)
    return max_abs_error


