"""Preprocess HistData MetaTrader EUR/USD M1 data into clean 5m returns."""

from multi_scale_volatility.preprocessing.cleaning import clean_1m
from multi_scale_volatility.preprocessing.constants import FIXED_EST, PRICE_COLUMNS, RAW_COLUMNS, UTC
from multi_scale_volatility.preprocessing.paths import PreprocessingPaths
from multi_scale_volatility.preprocessing.pipeline import run_preprocessing
from multi_scale_volatility.preprocessing.raw import discover_raw_csvs, load_raw_1m
from multi_scale_volatility.preprocessing.resampling import build_5m_ohlc
from multi_scale_volatility.preprocessing.returns import build_clean_returns

__all__ = [
    "FIXED_EST",
    "PRICE_COLUMNS",
    "PreprocessingPaths",
    "RAW_COLUMNS",
    "UTC",
    "build_5m_ohlc",
    "build_clean_returns",
    "clean_1m",
    "discover_raw_csvs",
    "load_raw_1m",
    "run_preprocessing",
]
