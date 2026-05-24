"""Create baseline return series from the final EUR/USD return dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.globals.columns import LOG_RETURN, TIMESTAMP_UTC
from src.globals.constants import GAUSSIAN_SEED, SHUFFLE_SEED
from src.globals.paths import (
    BASELINES_DIR,
    BASELINES_REPORT_JSON,
    FINAL_RETURNS_CSV,
    GAUSSIAN_RETURNS_CSV,
    SHUFFLE_RETURNS_CSV,
)
from src.utils.artifact_io import write_csv
from src.utils.json_utils import write_json


@dataclass(frozen=True)
class BaselinePaths:
    input_csv: Path = FINAL_RETURNS_CSV
    output_dir: Path = BASELINES_DIR

    @property
    def shuffle_csv(self) -> Path:
        return SHUFFLE_RETURNS_CSV if self.output_dir == BASELINES_DIR else (
            self.output_dir / SHUFFLE_RETURNS_CSV.name
        )

    @property
    def gaussian_csv(self) -> Path:
        return GAUSSIAN_RETURNS_CSV if self.output_dir == BASELINES_DIR else (
            self.output_dir / GAUSSIAN_RETURNS_CSV.name
        )

    @property
    def report_json(self) -> Path:
        return BASELINES_REPORT_JSON if self.output_dir == BASELINES_DIR else (
            self.output_dir / BASELINES_REPORT_JSON.name
        )


def create_baselines(
    paths: BaselinePaths | None = None,
    shuffle_seed: int = SHUFFLE_SEED,
    gaussian_seed: int = GAUSSIAN_SEED,
) -> dict[str, Any]:
    paths = paths or BaselinePaths()
    data = pd.read_csv(paths.input_csv, usecols=[TIMESTAMP_UTC, LOG_RETURN])
    if data.empty:
        raise ValueError(f"Input dataset is empty: {paths.input_csv}")

    timestamps = data[TIMESTAMP_UTC].copy()
    returns = data[LOG_RETURN].astype(float).to_numpy()
    empirical_mean = float(np.mean(returns))
    empirical_variance = float(np.var(returns, ddof=0))
    empirical_std = float(np.sqrt(empirical_variance))

    shuffle_rng = np.random.default_rng(shuffle_seed)
    shuffled_returns = shuffle_rng.permutation(returns)

    gaussian_rng = np.random.default_rng(gaussian_seed)
    gaussian_returns = gaussian_rng.normal(
        loc=0.0,
        scale=empirical_std,
        size=len(returns),
    )

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    shuffle_frame = pd.DataFrame(
        {TIMESTAMP_UTC: timestamps, LOG_RETURN: shuffled_returns}
    )
    gaussian_frame = pd.DataFrame(
        {TIMESTAMP_UTC: timestamps, LOG_RETURN: gaussian_returns}
    )
    write_csv(shuffle_frame, paths.shuffle_csv, index=False)
    write_csv(gaussian_frame, paths.gaussian_csv, index=False)

    report = {
        "input_csv": str(paths.input_csv),
        "input_rows": int(len(data)),
        "timestamp_start_utc": str(timestamps.iloc[0]),
        "timestamp_end_utc": str(timestamps.iloc[-1]),
        "empirical_mean_log_return": empirical_mean,
        "empirical_population_variance_log_return": empirical_variance,
        "empirical_population_std_log_return": empirical_std,
        "shuffle_baseline": {
            "seed": shuffle_seed,
            "output_csv": str(paths.shuffle_csv),
            "rows": int(len(shuffle_frame)),
            "mean_log_return": float(np.mean(shuffled_returns)),
            "population_variance_log_return": float(np.var(shuffled_returns, ddof=0)),
            "population_std_log_return": float(np.std(shuffled_returns, ddof=0)),
            "min_log_return": float(np.min(shuffled_returns)),
            "max_log_return": float(np.max(shuffled_returns)),
        },
        "gaussian_baseline": {
            "seed": gaussian_seed,
            "output_csv": str(paths.gaussian_csv),
            "rows": int(len(gaussian_frame)),
            "target_mean_log_return": 0.0,
            "target_population_variance_log_return": empirical_variance,
            "realized_mean_log_return": float(np.mean(gaussian_returns)),
            "realized_population_variance_log_return": float(
                np.var(gaussian_returns, ddof=0)
            ),
            "realized_population_std_log_return": float(
                np.std(gaussian_returns, ddof=0)
            ),
            "min_log_return": float(np.min(gaussian_returns)),
            "max_log_return": float(np.max(gaussian_returns)),
        },
    }
    write_json(paths.report_json, report)
    return report
