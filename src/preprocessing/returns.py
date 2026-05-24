"""Build clean return series from five-minute OHLC bars."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.columns import LOG_RETURN, PREVIOUS_TIMESTAMP_UTC, TIMESTAMP_UTC
from src.preprocessing.summary import int_key_counts
from src.utils.json_utils import json_scalar


def build_clean_returns(ohlc_5m: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns = ohlc_5m[[TIMESTAMP_UTC, "close", "n_m1"]].copy()
    returns = returns.sort_values(TIMESTAMP_UTC).reset_index(drop=True)
    returns[PREVIOUS_TIMESTAMP_UTC] = returns[TIMESTAMP_UTC].shift(1)
    returns["previous_close"] = returns["close"].shift(1)
    returns["delta_minutes"] = (
        (returns[TIMESTAMP_UTC] - returns[PREVIOUS_TIMESTAMP_UTC])
        .dt.total_seconds()
        .div(60)
    )
    returns[LOG_RETURN] = np.log(returns["close"]) - np.log(returns["previous_close"])
    returns = returns.dropna(
        subset=[PREVIOUS_TIMESTAMP_UTC, "previous_close", "delta_minutes"])

    clean_mask = returns["delta_minutes"].eq(5)
    clean_returns = returns.loc[clean_mask].copy()
    dropped_returns = returns.loc[~clean_mask].copy()

    clean_returns["delta_minutes"] = clean_returns["delta_minutes"].astype(int)
    dropped_returns["delta_minutes"] = dropped_returns["delta_minutes"].astype(int)

    clean_columns = [
        TIMESTAMP_UTC,
        "close",
        LOG_RETURN,
        PREVIOUS_TIMESTAMP_UTC,
        "previous_close",
        "delta_minutes",
        "n_m1",
    ]
    clean_returns = clean_returns[clean_columns].reset_index(drop=True)

    report = {
        "return_rows_clean": int(len(clean_returns)),
        "return_rows_dropped": int(len(dropped_returns)),
        "return_drop_rule": "delta_minutes != 5",
        "dropped_return_delta_minutes_counts": int_key_counts(
            dropped_returns["delta_minutes"]
        ),
        "dropped_returns": _dropped_returns_for_report(dropped_returns),
    }
    return clean_returns, report


def _dropped_returns_for_report(dropped: pd.DataFrame) -> list[dict[str, Any]]:
    if dropped.empty:
        return []

    rows = dropped.copy()
    rows["missing_5m_bars"] = rows["delta_minutes"].map(
        lambda minutes: max(int(minutes // 5) - 1, 0)
    )
    rows = rows[
        [
            TIMESTAMP_UTC,
            PREVIOUS_TIMESTAMP_UTC,
            "delta_minutes",
            "missing_5m_bars",
            "close",
            "previous_close",
            LOG_RETURN,
            "n_m1",
        ]
    ]

    records: list[dict[str, Any]] = []
    for record in rows.to_dict("records"):
        records.append({key: json_scalar(value) for key, value in record.items()})
    return records

