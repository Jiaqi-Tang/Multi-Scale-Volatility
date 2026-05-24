"""Build five-minute OHLC bars from cleaned one-minute data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.globals.columns import TIMESTAMP_UTC
from src.preprocessing.summary import int_key_counts
from src.utils.time_utils import iso_or_none


def build_5m_ohlc(clean_1m_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexed = clean_1m_frame.set_index(TIMESTAMP_UTC).sort_index()
    grouped = indexed.resample("5min", label="left", closed="left")

    ohlc = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        n_m1=("close", "count"),
        source_years=("source_year", lambda values: ",".join(
            map(str, sorted(set(values))))),
    )
    ohlc = ohlc[ohlc["n_m1"] > 0].reset_index()
    ohlc["complete"] = ohlc["n_m1"].eq(5)

    report = {
        "ohlc_5m_nonempty_rows": int(len(ohlc)),
        "ohlc_5m_complete_rows": int(ohlc["complete"].sum()),
        "ohlc_5m_partial_rows": int((~ohlc["complete"]).sum()),
        "ohlc_5m_n_m1_counts": int_key_counts(ohlc["n_m1"]),
        "ohlc_5m_first_timestamp_utc": iso_or_none(ohlc[TIMESTAMP_UTC].min()),
        "ohlc_5m_last_timestamp_utc": iso_or_none(ohlc[TIMESTAMP_UTC].max()),
    }
    return ohlc, report

