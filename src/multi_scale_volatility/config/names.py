"""Canonical names, labels, and scalar constants used across the project."""

from __future__ import annotations

from typing import Literal

ComponentType = Literal["detail", "approximation", "original"]
SeriesName = Literal["final", "shuffle", "gaussian"]

INDEX = "index"
TIMESTAMP_UTC = "timestamp_utc"
PREVIOUS_TIMESTAMP_UTC = "previous_timestamp_utc"
LOG_RETURN = "log_return"
COMPONENT = "component"
COMPONENT_TYPE = "component_type"
SERIES = "series"
ORIGINAL = "original"

SERIES_FINAL = "final"
SERIES_SHUFFLE = "shuffle"
SERIES_GAUSSIAN = "gaussian"
SERIES_ORDER = (SERIES_FINAL, SERIES_SHUFFLE, SERIES_GAUSSIAN)
BASELINE_SERIES = (SERIES_SHUFFLE, SERIES_GAUSSIAN)

DEFAULT_K = 11
BASE_INTERVAL_MINUTES = 5

TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 24
PERIODS_PER_HOUR = 60 // BASE_INTERVAL_MINUTES

SHUFFLE_SEED = 137
GAUSSIAN_SEED = 271

MONTE_CARLO_BASELINE_SIMULATIONS = 100
MONTE_CARLO_BASELINE_MASTER_SEED = 20260609
MONTE_CARLO_BASELINE_TYPES = ("shuffle", "gaussian")
MONTE_CARLO_BASELINE_QUANTILES = (0.05, 0.5, 0.95)
MONTE_CARLO_BASELINE_QUANTILE_METHOD = "linear"

K = "k"
SCALE_MINUTES = "scale_minutes"
SCALE_DAYS = "scale_days"

ENERGY = "energy"
RMS_VOLATILITY = "rms_volatility"
ANNUALIZED_RMS_VOLATILITY = "annualized_rms_volatility"
DETAIL_ENERGY_SHARE = "detail_energy_share"
TOTAL_COMPONENT_ENERGY_SHARE = "total_component_energy_share"

REPEAT_LENGTH = "repeat_length"
EFFECTIVE_N = "effective_n"
ORDINAL_WINDOWS = "ordinal_windows"
PERMUTATION_ENTROPY = "permutation_entropy"
NORMALIZED_ENTROPY = "normalized_entropy"
FINAL_NORMALIZED_ENTROPY = "final_normalized_entropy"
SHUFFLE_NORMALIZED_ENTROPY = "shuffle_normalized_entropy"
GAUSSIAN_NORMALIZED_ENTROPY = "gaussian_normalized_entropy"
ENTROPY_GAP_SHUFFLE = "entropy_gap_shuffle"
ENTROPY_GAP_GAUSSIAN = "entropy_gap_gaussian"

ORIGINAL_ENERGY = "original_energy"
DETAIL_ENERGY_SUM = "detail_energy_sum"
APPROXIMATION_ENERGY = "approximation_energy"
TOTAL_COMPONENT_ENERGY = "total_component_energy"
ENERGY_RECONSTRUCTION_GAP = "energy_reconstruction_gap"
DETAIL_ENERGY_SHARE_SUM = "detail_energy_share_sum"
TOTAL_COMPONENT_ENERGY_SHARE_SUM = "total_component_energy_share_sum"
