"""Preprocess HistData MetaTrader EUR/USD M1 data into clean 5m returns."""

from multi_scale_volatility.research.preprocessing.cleaning import clean_1m
from multi_scale_volatility.research.preprocessing.constants import FIXED_EST, PRICE_COLUMNS, RAW_COLUMNS, UTC
from multi_scale_volatility.research.preprocessing.paths import PreprocessingPaths
from multi_scale_volatility.research.preprocessing.pipeline import run_preprocessing
from multi_scale_volatility.research.preprocessing.raw import discover_raw_csvs, load_raw_1m
from multi_scale_volatility.research.preprocessing.resampling import build_5m_ohlc
from multi_scale_volatility.research.preprocessing.returns import build_clean_returns

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
