"""Artifact readers, writers, and serialization helpers."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.components import decomposition_components
from multi_scale_volatility.config.names import (
    COMPONENT,
    LOG_RETURN,
    SERIES_ORDER,
)
from multi_scale_volatility.config.schemas import (
    ENTROPY_GAP_COLUMNS,
    LAYER_ENTROPY_COLUMNS,
    VOLATILITY_COLUMNS,
    decomposition_columns,
)
from multi_scale_volatility.utils.validation import require_columns


def read_returns(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, usecols=[LOG_RETURN])
    if frame.empty:
        raise ValueError(f"Return file is empty: {path}")
    return frame[LOG_RETURN].astype(float).to_numpy()


def read_decomposition(path: Path, k: int) -> pd.DataFrame:
    columns = list(
        decomposition_columns(
            decomposition_components(k, include_original=False),
            include_timestamp=False,
        )
    )
    frame = pd.read_csv(path, usecols=columns)
    require_columns(frame, columns, path)
    return frame


def read_volatility(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, VOLATILITY_COLUMNS, path)
    validate_components(frame, path, k)
    return frame


def read_layer_entropy(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, LAYER_ENTROPY_COLUMNS, path)
    validate_components(frame, path, k)
    return frame


def read_entropy_gaps(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, ENTROPY_GAP_COLUMNS, path)
    validate_components(frame, path, k)
    return frame


def read_entropy_pattern_counts(path: Path, k: int) -> dict[str, dict[str, dict[str, int]]]:
    report = read_json(path)
    pattern_counts = report.get("pattern_counts")
    if not isinstance(pattern_counts, dict):
        raise ValueError(f"Missing pattern_counts in {path}")

    expected_components = decomposition_components(k, include_original=False)
    output: dict[str, dict[str, dict[str, int]]] = {}
    for series in SERIES_ORDER:
        series_counts = pattern_counts.get(series)
        if not isinstance(series_counts, dict):
            raise ValueError(f"Missing pattern counts for series {series} in {path}")

        output[series] = {}
        for component in expected_components:
            component_counts = series_counts.get(component)
            if not isinstance(component_counts, dict):
                raise ValueError(f"Missing pattern counts for {series} {component} in {path}")
            output[series][component] = validate_pattern_counts(
                component_counts,
                path,
                series,
                component,
            )
    return output


def read_report(path: Path) -> dict[str, Any]:
    return read_json(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path, **to_csv_kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        frame.to_csv(temp_path, **to_csv_kwargs)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_parquet(frame: pd.DataFrame, path: Path, **to_parquet_kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        frame.to_parquet(temp_path, **to_parquet_kwargs)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        temp_path.write_text(text, encoding=encoding)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def json_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def validate_components(frame: pd.DataFrame, path: Path, k: int) -> None:
    expected_components = set(decomposition_components(k, include_original=False))
    unexpected_components = sorted(set(frame[COMPONENT]).difference(expected_components))
    if unexpected_components:
        raise ValueError(f"Unexpected components in {path}: {unexpected_components}")


def validate_pattern_counts(
    counts: dict[str, Any],
    path: Path,
    series: str,
    component: str,
) -> dict[str, int]:
    output: dict[str, int] = {}
    for pattern, count in counts.items():
        if not isinstance(pattern, str):
            raise ValueError(f"Non-string pattern for {series} {component} in {path}")
        if not isinstance(count, int):
            raise ValueError(
                f"Non-integer count for {series} {component} pattern {pattern} in {path}"
            )
        if count < 0:
            raise ValueError(
                f"Negative count for {series} {component} pattern {pattern} in {path}"
            )
        output[pattern] = count
    if not output:
        raise ValueError(f"Empty pattern counts for {series} {component} in {path}")
    return output


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)
