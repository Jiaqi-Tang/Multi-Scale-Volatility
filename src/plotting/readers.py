"""CSV readers and schema checks for plotting inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.globals.columns import COMPONENT, COMPONENT_TYPE, INDEX, LOG_RETURN, ORIGINAL, SERIES
from src.globals.series import SERIES_ORDER
from src.scale_utils import decomposition_components
from src.utils.json_utils import read_json
from src.utils.validation import require_columns


def read_returns(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, usecols=[LOG_RETURN])
    if frame.empty:
        raise ValueError(f"Return file is empty: {path}")
    return frame[LOG_RETURN].astype(float).to_numpy()


def read_decomposition(path: Path, k: int) -> pd.DataFrame:
    columns = [INDEX, ORIGINAL] + decomposition_components(k, include_original=False)
    frame = pd.read_csv(path, usecols=columns)
    require_columns(frame, columns, path)
    return frame


def read_volatility(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        SERIES,
        COMPONENT,
        COMPONENT_TYPE,
        "detail_energy_share",
        "total_component_energy_share",
        "rms_volatility",
        "annualized_rms_volatility",
    }
    require_columns(frame, required_columns, path)
    validate_components(frame, path, k)
    return frame


def read_layer_entropy(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        SERIES,
        COMPONENT,
        "permutation_entropy",
        "normalized_entropy",
    }
    require_columns(frame, required_columns, path)
    validate_components(frame, path, k)
    return frame


def read_entropy_gaps(path: Path, k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        COMPONENT,
        "entropy_gap_shuffle",
        "entropy_gap_gaussian",
    }
    require_columns(frame, required_columns, path)
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
                raise ValueError(
                    f"Missing pattern counts for {series} {component} in {path}"
                )
            output[series][component] = _validate_pattern_counts(
                component_counts,
                path,
                series,
                component,
            )
    return output


def validate_components(frame: pd.DataFrame, path: Path, k: int) -> None:
    expected_components = set(decomposition_components(k, include_original=False))
    unexpected_components = sorted(set(frame[COMPONENT]).difference(expected_components))
    if unexpected_components:
        raise ValueError(f"Unexpected components in {path}: {unexpected_components}")


def _validate_pattern_counts(
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
