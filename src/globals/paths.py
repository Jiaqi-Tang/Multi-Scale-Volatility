"""Default filesystem paths for pipeline artifacts."""

from pathlib import Path

RAW_METATRADER_DIR = Path("data/raw/metatrader")

INTERMEDIATE_DIR = Path("data/intermediate")
CLEAN_1M_CSV = INTERMEDIATE_DIR / "eurusd_1m_utc_clean.csv"
OHLC_5M_CSV = INTERMEDIATE_DIR / "eurusd_5m_ohlc_utc_nonempty.csv"
CLEAN_RETURNS_CSV = INTERMEDIATE_DIR / "eurusd_5m_log_returns_clean.csv"
PREPROCESSING_REPORT_JSON = INTERMEDIATE_DIR / "preprocessing_report.json"

FINAL_DIR = Path("data/final")
FINAL_RETURNS_CSV = FINAL_DIR / "eurusd_5m_log_returns_final.csv"
TRUNCATION_REPORT_JSON = FINAL_DIR / "truncation_report.json"

BASELINES_DIR = Path("data/baselines")
SHUFFLE_RETURNS_CSV = BASELINES_DIR / "eurusd_5m_log_returns_shuffle.csv"
GAUSSIAN_RETURNS_CSV = BASELINES_DIR / "eurusd_5m_log_returns_gaussian.csv"
BASELINES_REPORT_JSON = BASELINES_DIR / "baselines_report.json"

DECOMPOSITION_DIR = Path("data/decomposition")
FINAL_DECOMPOSITION_CSV = DECOMPOSITION_DIR / "final_decomposition.csv"
SHUFFLE_DECOMPOSITION_CSV = DECOMPOSITION_DIR / "shuffle_decomposition.csv"
GAUSSIAN_DECOMPOSITION_CSV = DECOMPOSITION_DIR / "gaussian_decomposition.csv"
DECOMPOSITION_REPORT_JSON = DECOMPOSITION_DIR / "decomposition_report.json"

VOLATILITY_RESULTS_DIR = Path("results/volatility")
VOLATILITY_CSV = VOLATILITY_RESULTS_DIR / "layer_volatility.csv"
VOLATILITY_REPORT_JSON = VOLATILITY_RESULTS_DIR / "volatility_report.json"

ENTROPY_RESULTS_DIR = Path("results/entropy")
LAYER_ENTROPY_CSV = ENTROPY_RESULTS_DIR / "layer_entropy.csv"
ENTROPY_GAPS_CSV = ENTROPY_RESULTS_DIR / "entropy_gaps.csv"
ENTROPY_REPORT_JSON = ENTROPY_RESULTS_DIR / "entropy_report.json"

EDA_PLOTS_DIR = Path("plots/eda/returns")
DECOMPOSITION_PLOTS_DIR = Path("plots/eda/decomposition")
VOLATILITY_PLOTS_DIR = Path("plots/results/volatility")
ENTROPY_PLOTS_DIR = Path("plots/results/entropy")
MEMO_PLOTS_DIR = Path("plots/memo")

