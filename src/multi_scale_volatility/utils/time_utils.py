"""Timestamp formatting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def iso_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    return str(value)

