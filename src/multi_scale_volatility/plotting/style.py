"""Shared plotting style constants."""

from __future__ import annotations

import matplotlib

from multi_scale_volatility.config.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_SHUFFLE

matplotlib.use("Agg")

FIGURE_DPI = 180
GRID_FIGURE_DPI = 160

FINAL_COLOR = "#2f6f9f"
SHUFFLE_COLOR = "#5c9f52"
GAUSSIAN_COLOR = "#c76d3b"
FINAL_DARK_COLOR = "#17486b"
GAUSSIAN_DARK_COLOR = "#8d3f19"

SERIES_COLORS = {
    SERIES_FINAL: FINAL_COLOR,
    SERIES_SHUFFLE: SHUFFLE_COLOR,
    SERIES_GAUSSIAN: GAUSSIAN_COLOR,
}

SERIES_LABELS = {
    SERIES_FINAL: "EUR/USD final",
    SERIES_SHUFFLE: "Shuffled baseline",
    SERIES_GAUSSIAN: "Gaussian baseline",
}
