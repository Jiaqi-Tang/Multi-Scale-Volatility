"""Correlation helpers for decomposition components."""

from __future__ import annotations

import pandas as pd


def absolute_component_correlation(
    frame: pd.DataFrame,
    components: list[str],
) -> pd.DataFrame:
    return frame[components].abs().corr(method="pearson")


def absolute_component_correlation_difference(
    final_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    components: list[str],
) -> pd.DataFrame:
    return (
        absolute_component_correlation(final_frame, components)
        - absolute_component_correlation(baseline_frame, components)
    )

