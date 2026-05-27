"""Distribution helpers."""

from __future__ import annotations

import statistics

import numpy as np


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def normal_quantiles_for_values(
    values: np.ndarray,
    quantile_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    std = float(np.sqrt(np.var(values, ddof=0)))
    probabilities = np.linspace(
        1.0 / (quantile_points + 1),
        quantile_points / (quantile_points + 1),
        quantile_points,
    )
    empirical_quantiles = np.quantile(values, probabilities)
    gaussian_quantiles = np.array(
        [statistics.NormalDist(mu=0.0, sigma=std).inv_cdf(p) for p in probabilities]
    )
    return gaussian_quantiles, empirical_quantiles

