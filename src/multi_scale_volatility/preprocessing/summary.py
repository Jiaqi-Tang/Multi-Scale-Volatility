"""Report summary helpers for preprocessing."""

from __future__ import annotations

import pandas as pd


def int_key_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = series.astype(int).value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}

