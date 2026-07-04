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

MONTE_CARLO_BASELINES_DATA_DIR = Path("data/monte_carlo_baselines")
MONTE_CARLO_BASELINE_RETURNS_DIR = MONTE_CARLO_BASELINES_DATA_DIR / "returns"
MONTE_CARLO_BASELINE_DECOMPOSITION_DIR = (
    MONTE_CARLO_BASELINES_DATA_DIR / "decomposition"
)

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

ROLLING_RESULTS_DIR = Path("results/rolling")
ROLLING_WINDOW_METADATA_CSV = ROLLING_RESULTS_DIR / "rolling_window_metadata.csv"
ROLLING_LAYER_VOLATILITY_CSV = ROLLING_RESULTS_DIR / "rolling_layer_volatility.csv"
ROLLING_WINDOW_SUMMARY_CSV = ROLLING_RESULTS_DIR / "rolling_window_summary.csv"
ROLLING_SCALE_GROUP_SUMMARY_CSV = ROLLING_RESULTS_DIR / "rolling_scale_group_summary.csv"
ROLLING_EXAMPLE_WINDOWS_CSV = ROLLING_RESULTS_DIR / "rolling_example_windows.csv"
ROLLING_REPORT_JSON = ROLLING_RESULTS_DIR / "rolling_report.json"

ROLLING_BASELINE_RESULTS_DIR = Path("results/rolling_baselines")
ROLLING_BASELINE_CORRELATION_SIMULATIONS_CSV = (
    ROLLING_BASELINE_RESULTS_DIR / "rolling_correlation_simulations.csv"
)
ROLLING_BASELINE_CORRELATION_SUMMARY_CSV = (
    ROLLING_BASELINE_RESULTS_DIR / "rolling_correlation_summary.csv"
)
ROLLING_BASELINE_CORRELATION_EMPIRICAL_COMPARISON_CSV = (
    ROLLING_BASELINE_RESULTS_DIR / "rolling_correlation_empirical_comparison.csv"
)
ROLLING_BASELINE_RUNTIME_LOG_CSV = ROLLING_BASELINE_RESULTS_DIR / "runtime_log.csv"
ROLLING_BASELINE_REPORT_JSON = ROLLING_BASELINE_RESULTS_DIR / "rolling_baseline_report.json"

MONTE_CARLO_BASELINES_RESULTS_DIR = Path("results/monte_carlo_baselines")
MONTE_CARLO_BASELINE_CONFIG_JSON = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "monte_carlo_config.json"
)
MONTE_CARLO_BASELINE_AUDIT_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "baseline_simulation_audit.csv"
)
MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "runtime_log.csv"
)
MC_LAYER_VOLATILITY_SIMULATIONS_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "layer_volatility_simulations.csv"
)
MC_LAYER_VOLATILITY_SUMMARY_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "layer_volatility_summary.csv"
)
MC_LAYER_VOLATILITY_EMPIRICAL_COMPARISON_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "layer_volatility_empirical_comparison.csv"
)
MC_LAYER_ENTROPY_SIMULATIONS_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "layer_entropy_simulations.csv"
)
MC_LAYER_ENTROPY_SUMMARY_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "layer_entropy_summary.csv"
)
MC_LAYER_ENTROPY_EMPIRICAL_COMPARISON_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "layer_entropy_empirical_comparison.csv"
)
MC_ACF_SIMULATIONS_CSV = MONTE_CARLO_BASELINES_RESULTS_DIR / "acf_simulations.csv"
MC_ACF_SUMMARY_CSV = MONTE_CARLO_BASELINES_RESULTS_DIR / "acf_summary.csv"
MC_ACF_EMPIRICAL_COMPARISON_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "acf_empirical_comparison.csv"
)
MC_COMPONENT_ACF_SIMULATIONS_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "component_acf_simulations.csv"
)
MC_COMPONENT_ACF_SUMMARY_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "component_acf_summary.csv"
)
MC_COMPONENT_ACF_EMPIRICAL_COMPARISON_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "component_acf_empirical_comparison.csv"
)
MC_ABS_COMPONENT_CORRELATION_SIMULATIONS_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "abs_component_correlation_simulations.csv"
)
MC_ABS_COMPONENT_CORRELATION_SUMMARY_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR / "abs_component_correlation_summary.csv"
)
MC_ABS_COMPONENT_CORRELATION_EMPIRICAL_COMPARISON_CSV = (
    MONTE_CARLO_BASELINES_RESULTS_DIR
    / "abs_component_correlation_empirical_comparison.csv"
)

MEMO_PLOTS_DIR = Path("plots/memo")
DATA_EDA_RETURNS_PLOTS_DIR = Path("plots/results/data_eda/returns")
DATA_EDA_DECOMPOSITION_PLOTS_DIR = Path("plots/results/data_eda/decomposition")
GLOBAL_VOLATILITY_PLOTS_DIR = Path("plots/results/global_data/volatility")
GLOBAL_ENTROPY_PLOTS_DIR = Path("plots/results/global_data/entropy")
GLOBAL_CORRELATION_PLOTS_DIR = Path("plots/results/global_data/correlation")
ROLLING_WINDOWS_PLOTS_DIR = Path("plots/results/rolling_windows")
ROLLING_BASELINE_PLOTS_DIR = ROLLING_WINDOWS_PLOTS_DIR / "baselines"
