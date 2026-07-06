"""Small validation helpers shared by pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def require_positive_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def require_non_negative_k(k: int) -> None:
    if k < 0:
        raise ValueError("k must be non-negative")


def require_non_empty_frame(frame: pd.DataFrame, path: Path | str) -> None:
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    path: Path | str,
) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")


def require_finite_array(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")

