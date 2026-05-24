"""Preprocess HistData MetaTrader EUR/USD M1 data into clean 5m returns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.globals.columns import LOG_RETURN, PREVIOUS_TIMESTAMP_UTC, TIMESTAMP_UTC
from src.globals.paths import (
    CLEAN_1M_CSV,
    CLEAN_RETURNS_CSV,
    INTERMEDIATE_DIR,
    OHLC_5M_CSV,
    PREPROCESSING_REPORT_JSON,
    RAW_METATRADER_DIR,
)
from src.utils.artifact_io import write_csv
from src.utils.json_utils import json_scalar, write_json
from src.utils.time_utils import iso_or_none


RAW_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
FIXED_EST = timezone(timedelta(hours=-5))
UTC = timezone.utc


@dataclass(frozen=True)
class PreprocessingPaths:
    raw_dir: Path = RAW_METATRADER_DIR
    intermediate_dir: Path = INTERMEDIATE_DIR

    @property
    def clean_1m_csv(self) -> Path:
        return CLEAN_1M_CSV if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / CLEAN_1M_CSV.name
        )

    @property
    def ohlc_5m_csv(self) -> Path:
        return OHLC_5M_CSV if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / OHLC_5M_CSV.name
        )

    @property
    def report_json(self) -> Path:
        return PREPROCESSING_REPORT_JSON if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / PREPROCESSING_REPORT_JSON.name
        )

    @property
    def clean_returns_csv(self) -> Path:
        return CLEAN_RETURNS_CSV if self.intermediate_dir == INTERMEDIATE_DIR else (
            self.intermediate_dir / CLEAN_RETURNS_CSV.name
        )


def discover_raw_csvs(raw_dir: Path) -> list[Path]:
    paths = sorted(raw_dir.rglob("DAT_MT_EURUSD_M1_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No raw MetaTrader CSV files found under {raw_dir}")
    return paths


def _source_year(path: Path) -> int:
    match = re.search(r"_(\d{4})\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not infer source year from {path}")
    return int(match.group(1))


def load_raw_1m(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    row_counts: dict[str, int] = {}

    for path in discover_raw_csvs(raw_dir):
        frame = pd.read_csv(path, header=None, names=RAW_COLUMNS)
        frame["source_year"] = _source_year(path)
        frame["source_file"] = str(path)
        row_counts[path.name] = int(len(frame))
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    report = {
        "raw_files": row_counts,
        "raw_rows_loaded": int(len(raw)),
    }
    return raw, report


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
        "ohlc_5m_n_m1_counts": _int_key_counts(ohlc["n_m1"]),
        "ohlc_5m_first_timestamp_utc": iso_or_none(ohlc[TIMESTAMP_UTC].min()),
        "ohlc_5m_last_timestamp_utc": iso_or_none(ohlc[TIMESTAMP_UTC].max()),
    }
    return ohlc, report


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
    returns[LOG_RETURN] = np.log(
        returns["close"]) - np.log(returns["previous_close"])
    returns = returns.dropna(
        subset=[PREVIOUS_TIMESTAMP_UTC, "previous_close", "delta_minutes"])

    clean_mask = returns["delta_minutes"].eq(5)
    clean_returns = returns.loc[clean_mask].copy()
    dropped_returns = returns.loc[~clean_mask].copy()

    clean_returns["delta_minutes"] = clean_returns["delta_minutes"].astype(int)
    dropped_returns["delta_minutes"] = dropped_returns["delta_minutes"].astype(
        int)

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
        "dropped_return_delta_minutes_counts": _int_key_counts(dropped_returns["delta_minutes"]),
        "dropped_returns": _dropped_returns_for_report(dropped_returns),
    }
    return clean_returns, report


def run_preprocessing(paths: PreprocessingPaths | None = None) -> dict[str, Any]:
    paths = paths or PreprocessingPaths()
    paths.intermediate_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "asset": "EUR/USD",
        "raw_frequency": "1min",
        "output_bar_frequency": "5min",
        "raw_timezone_assumption": "fixed EST (UTC-05:00), no DST",
        "output_timezone": "UTC",
    }

    raw, raw_report = load_raw_1m(paths.raw_dir)
    report.update(raw_report)

    clean_1m_frame, clean_1m_report = clean_1m(raw)
    report.update(clean_1m_report)
    write_csv(clean_1m_frame, paths.clean_1m_csv, index=False)

    ohlc_5m, ohlc_report = build_5m_ohlc(clean_1m_frame)
    report.update(ohlc_report)
    write_csv(ohlc_5m, paths.ohlc_5m_csv, index=False)

    clean_returns, returns_report = build_clean_returns(ohlc_5m)
    report.update(returns_report)
    write_csv(clean_returns, paths.clean_returns_csv, index=False)

    report["outputs"] = {
        "clean_1m_csv": str(paths.clean_1m_csv),
        "ohlc_5m_csv": str(paths.ohlc_5m_csv),
        "clean_returns_csv": str(paths.clean_returns_csv),
        "report_json": str(paths.report_json),
    }
    write_json(paths.report_json, report)
    return report


def _int_key_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = series.astype(int).value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}


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
        records.append(
            {
                key: json_scalar(value)
                for key, value in record.items()
            }
        )
    return records
