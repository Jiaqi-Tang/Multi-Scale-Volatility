"""Preprocessing pipeline orchestration."""

from __future__ import annotations

from typing import Any

from multi_scale_volatility.preprocessing.cleaning import clean_1m
from multi_scale_volatility.preprocessing.paths import PreprocessingPaths
from multi_scale_volatility.preprocessing.raw import load_raw_1m
from multi_scale_volatility.preprocessing.resampling import build_5m_ohlc
from multi_scale_volatility.preprocessing.returns import build_clean_returns
from multi_scale_volatility.utils.artifact_io import write_csv
from multi_scale_volatility.utils.json_utils import write_json


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
