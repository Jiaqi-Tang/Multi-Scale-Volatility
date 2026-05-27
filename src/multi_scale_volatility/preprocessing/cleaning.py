"""Raw one-minute OHLC cleaning."""

from __future__ import annotations

from typing import Any

import pandas as pd

from multi_scale_volatility.config.columns import TIMESTAMP_UTC
from multi_scale_volatility.preprocessing.constants import FIXED_EST, PRICE_COLUMNS, UTC
from multi_scale_volatility.utils.time_utils import iso_or_none


def clean_1m(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = raw.copy()
    raw["timestamp_raw"] = pd.to_datetime(
        raw["date"] + " " + raw["time"],
        format="%Y.%m.%d %H:%M",
        errors="raise",
    )

    for column in PRICE_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="raise")

    invalid_ohlc = raw[
        (raw["low"] > raw[["open", "close"]].min(axis=1))
        | (raw["high"] < raw[["open", "close"]].max(axis=1))
        | (raw["high"] < raw["low"])
    ]
    if not invalid_ohlc.empty:
        raise ValueError(f"Found {len(invalid_ohlc)} invalid OHLC rows")

    exact_subset = ["timestamp_raw", *PRICE_COLUMNS, "volume", "source_year"]
    exact_duplicates = int(raw.duplicated(
        subset=exact_subset, keep="first").sum())
    clean = raw.drop_duplicates(subset=exact_subset, keep="first").copy()

    differing_duplicate_timestamps = _find_differing_duplicate_timestamps(
        clean)
    if differing_duplicate_timestamps:
        raise ValueError(
            "Found duplicate timestamps with different OHLC/volume values. "
            f"Examples: {differing_duplicate_timestamps[:5]}"
        )

    clean[TIMESTAMP_UTC] = (
        clean["timestamp_raw"].dt.tz_localize(FIXED_EST).dt.tz_convert(UTC)
    )
    clean = clean.sort_values(TIMESTAMP_UTC).reset_index(drop=True)

    output_columns = [
        TIMESTAMP_UTC,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_year",
    ]
    clean = clean[output_columns]

    report = {
        "raw_exact_duplicate_rows_dropped": exact_duplicates,
        "clean_1m_rows": int(len(clean)),
        "clean_1m_first_timestamp_utc": iso_or_none(clean[TIMESTAMP_UTC].min()),
        "clean_1m_last_timestamp_utc": iso_or_none(clean[TIMESTAMP_UTC].max()),
    }
    return clean, report


def _find_differing_duplicate_timestamps(frame: pd.DataFrame) -> list[dict[str, Any]]:
    counts = frame.groupby("timestamp_raw").size()
    duplicate_timestamps = counts[counts > 1].index
    if duplicate_timestamps.empty:
        return []

    examples: list[dict[str, Any]] = []
    for timestamp in duplicate_timestamps:
        rows = frame.loc[
            frame["timestamp_raw"].eq(timestamp),
            ["timestamp_raw", *PRICE_COLUMNS, "volume"],
        ].drop_duplicates()
        if len(rows) > 1:
            examples.append(
                {
                    "timestamp_raw": timestamp.isoformat(),
                    "rows": rows.astype({"timestamp_raw": str}).to_dict("records"),
                }
            )
    return examples
