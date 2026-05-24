"""Standardize EUR/USD return-series length for dyadic decompositions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.columns import LOG_RETURN, PREVIOUS_TIMESTAMP_UTC, TIMESTAMP_UTC
from src.config.constants import BASE_INTERVAL_MINUTES, DEFAULT_K
from src.config.paths import CLEAN_RETURNS_CSV, FINAL_RETURNS_CSV, TRUNCATION_REPORT_JSON
from src.utils.artifact_io import write_csv
from src.utils.json_utils import write_json
from src.utils.time_utils import iso_or_none
from src.utils.validation import require_non_negative_k


@dataclass(frozen=True)
class LengthStandardizationPaths:
    input_csv: Path = CLEAN_RETURNS_CSV
    output_csv: Path = FINAL_RETURNS_CSV
    report_json: Path = TRUNCATION_REPORT_JSON


def standardize_length(
    paths: LengthStandardizationPaths | None = None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Trim clean returns from the end so length is divisible by 2**k."""
    paths = paths or LengthStandardizationPaths()
    require_non_negative_k(k)

    data = pd.read_csv(paths.input_csv, parse_dates=[TIMESTAMP_UTC, PREVIOUS_TIMESTAMP_UTC])
    if data.empty:
        raise ValueError(f"Input dataset is empty: {paths.input_csv}")
    if LOG_RETURN not in data.columns:
        raise ValueError(f"Input dataset must contain a {LOG_RETURN} column")

    block_size = 2**k
    input_rows = len(data)
    truncated_rows = (input_rows // block_size) * block_size
    if truncated_rows == 0:
        raise ValueError(
            f"Input rows ({input_rows}) are fewer than block size 2**{k} ({block_size})"
        )

    final = data.iloc[:truncated_rows].copy()
    dropped_tail = data.iloc[truncated_rows:].copy()

    write_csv(final, paths.output_csv, index=False)

    returns = final[LOG_RETURN]
    report = {
        "input_csv": str(paths.input_csv),
        "output_csv": str(paths.output_csv),
        "K": k,
        "block_size": block_size,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "block_span_minutes": block_size * BASE_INTERVAL_MINUTES,
        "block_span_days": block_size * BASE_INTERVAL_MINUTES / (60 * 24),
        "input_rows": int(input_rows),
        "truncated_rows": int(truncated_rows),
        "dropped_tail_rows": int(len(dropped_tail)),
        "input_start_timestamp_utc": iso_or_none(data[TIMESTAMP_UTC].iloc[0]),
        "input_end_timestamp_utc": iso_or_none(data[TIMESTAMP_UTC].iloc[-1]),
        "truncated_start_timestamp_utc": iso_or_none(final[TIMESTAMP_UTC].iloc[0]),
        "truncated_end_timestamp_utc": iso_or_none(final[TIMESTAMP_UTC].iloc[-1]),
        "dropped_tail_start_timestamp_utc": iso_or_none(
            dropped_tail[TIMESTAMP_UTC].iloc[0] if not dropped_tail.empty else None
        ),
        "dropped_tail_end_timestamp_utc": iso_or_none(
            dropped_tail[TIMESTAMP_UTC].iloc[-1] if not dropped_tail.empty else None
        ),
        "mean_log_return": float(returns.mean()),
        "variance_log_return": float(returns.var(ddof=0)),
        "std_log_return": float(returns.std(ddof=0)),
        "min_log_return": float(returns.min()),
        "max_log_return": float(returns.max()),
        "median_log_return": float(returns.median()),
        "skewness_log_return": float(returns.skew()),
        "kurtosis_log_return": float(returns.kurt()),
    }

    write_json(paths.report_json, report)
    return report
