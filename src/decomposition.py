"""Block-average multi-scale decomposition for return series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.globals.columns import INDEX, LOG_RETURN, ORIGINAL, TIMESTAMP_UTC
from src.globals.constants import BASE_INTERVAL_MINUTES, DEFAULT_K
from src.globals.paths import (
    DECOMPOSITION_DIR,
    DECOMPOSITION_REPORT_JSON,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    GAUSSIAN_RETURNS_CSV,
    SHUFFLE_DECOMPOSITION_CSV,
    SHUFFLE_RETURNS_CSV,
)
from src.globals.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_SHUFFLE
from src.utils.artifact_io import write_csv
from src.utils.json_utils import write_json
from src.utils.validation import require_finite_array, require_positive_k

RECONSTRUCTION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DecompositionInput:
    name: str
    input_csv: Path
    output_csv: Path


@dataclass(frozen=True)
class DecompositionPaths:
    final_csv: Path = FINAL_RETURNS_CSV
    shuffle_csv: Path = SHUFFLE_RETURNS_CSV
    gaussian_csv: Path = GAUSSIAN_RETURNS_CSV
    output_dir: Path = DECOMPOSITION_DIR

    @property
    def report_json(self) -> Path:
        return DECOMPOSITION_REPORT_JSON if self.output_dir == DECOMPOSITION_DIR else (
            self.output_dir / DECOMPOSITION_REPORT_JSON.name
        )

    def inputs(self) -> list[DecompositionInput]:
        return [
            DecompositionInput(
                name=SERIES_FINAL,
                input_csv=self.final_csv,
                output_csv=(
                    FINAL_DECOMPOSITION_CSV
                    if self.output_dir == DECOMPOSITION_DIR
                    else self.output_dir / FINAL_DECOMPOSITION_CSV.name
                ),
            ),
            DecompositionInput(
                name=SERIES_SHUFFLE,
                input_csv=self.shuffle_csv,
                output_csv=(
                    SHUFFLE_DECOMPOSITION_CSV
                    if self.output_dir == DECOMPOSITION_DIR
                    else self.output_dir / SHUFFLE_DECOMPOSITION_CSV.name
                ),
            ),
            DecompositionInput(
                name=SERIES_GAUSSIAN,
                input_csv=self.gaussian_csv,
                output_csv=(
                    GAUSSIAN_DECOMPOSITION_CSV
                    if self.output_dir == DECOMPOSITION_DIR
                    else self.output_dir / GAUSSIAN_DECOMPOSITION_CSV.name
                ),
            ),
        ]


def run_decomposition(
    paths: DecompositionPaths | None = None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    paths = paths or DecompositionPaths()
    require_positive_k(k)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "K": k,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "block_size_max": 2**k,
        "max_scale_minutes": BASE_INTERVAL_MINUTES * (2**k),
        "max_scale_days": BASE_INTERVAL_MINUTES * (2**k) / (60 * 24),
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "series": {},
    }

    for item in paths.inputs():
        report["series"][item.name] = decompose_csv(item, k=k)

    write_json(paths.report_json, report)
    return report


def decompose_csv(item: DecompositionInput, k: int) -> dict[str, Any]:
    frame = pd.read_csv(item.input_csv, usecols=[TIMESTAMP_UTC, LOG_RETURN])
    if frame.empty:
        raise ValueError(f"Input dataset is empty: {item.input_csv}")
    if frame[LOG_RETURN].isna().any():
        raise ValueError(f"Input contains NaN {LOG_RETURN} values: {item.input_csv}")

    values = frame[LOG_RETURN].astype(float).to_numpy()
    require_finite_array(values, f"Input {LOG_RETURN} values in {item.input_csv}")

    n = len(values)
    block_size_max = 2**k
    if n % block_size_max != 0:
        raise ValueError(
            f"Input length {n} is not divisible by 2**{k} ({block_size_max}): "
            f"{item.input_csv}"
        )

    details, final_approximation = decompose_values(values, k=k)
    reconstruction = final_approximation.copy()
    for detail in details:
        reconstruction += detail
    error = values - reconstruction
    max_abs_error = float(np.max(np.abs(error)))
    mean_abs_error = float(np.mean(np.abs(error)))
    if max_abs_error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"Reconstruction error {max_abs_error} exceeds tolerance "
            f"{RECONSTRUCTION_TOLERANCE} for {item.name}"
        )

    output = pd.DataFrame(
        {
            "index": np.arange(n, dtype=np.int64),
            TIMESTAMP_UTC: frame[TIMESTAMP_UTC],
            ORIGINAL: values,
        }
    )
    for scale, detail in enumerate(details, start=1):
        output[f"D_{scale:02d}"] = detail
    output[f"A_{k:02d}"] = final_approximation

    write_csv(output, item.output_csv, index=False)

    return {
        "input_csv": str(item.input_csv),
        "output_csv": str(item.output_csv),
        "N": int(n),
        "K": int(k),
        "block_size_max": int(block_size_max),
        "max_scale_minutes": int(BASE_INTERVAL_MINUTES * block_size_max),
        "max_scale_days": BASE_INTERVAL_MINUTES * block_size_max / (60 * 24),
        "timestamp_start_utc": str(frame[TIMESTAMP_UTC].iloc[0]),
        "timestamp_end_utc": str(frame[TIMESTAMP_UTC].iloc[-1]),
        "max_abs_reconstruction_error": max_abs_error,
        "mean_abs_reconstruction_error": mean_abs_error,
    }


def decompose_values(values: np.ndarray, k: int) -> tuple[list[np.ndarray], np.ndarray]:
    approximations = [values.astype(float, copy=True)]
    for scale in range(1, k + 1):
        block_size = 2**scale
        approximations.append(_expanded_block_mean(values, block_size))

    details = [
        approximations[scale - 1] - approximations[scale]
        for scale in range(1, k + 1)
    ]
    return details, approximations[k]


def _expanded_block_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    if len(values) % block_size != 0:
        raise ValueError(
            f"Series length {len(values)} is not divisible by block size {block_size}"
        )
    block_means = values.reshape(-1, block_size).mean(axis=1)
    return np.repeat(block_means, block_size)
