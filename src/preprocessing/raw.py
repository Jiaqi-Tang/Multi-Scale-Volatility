"""Raw MetaTrader CSV discovery and loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.constants import RAW_COLUMNS


def discover_raw_csvs(raw_dir: Path) -> list[Path]:
    paths = sorted(raw_dir.rglob("DAT_MT_EURUSD_M1_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No raw MetaTrader CSV files found under {raw_dir}")
    return paths


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


def _source_year(path: Path) -> int:
    match = re.search(r"_(\d{4})\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not infer source year from {path}")
    return int(match.group(1))

