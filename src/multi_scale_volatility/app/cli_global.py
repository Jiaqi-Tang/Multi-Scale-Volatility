"""Global-diagnosis command-line commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from multi_scale_volatility.app.cli_common import print_json
from multi_scale_volatility.core.config.names import DEFAULT_K
from multi_scale_volatility.core.config.paths import (
    CLEAN_RETURNS_CSV,
    DECOMPOSITION_DIR,
    ENTROPY_RESULTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    GAUSSIAN_RETURNS_CSV,
    INTERMEDIATE_DIR,
    LAYER_ENTROPY_CSV,
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV,
    MONTE_CARLO_BASELINES_DATA_DIR,
    MONTE_CARLO_BASELINES_RESULTS_DIR,
    RAW_METATRADER_DIR,
    SHUFFLE_DECOMPOSITION_CSV,
    SHUFFLE_RETURNS_CSV,
    TRUNCATION_REPORT_JSON,
    VOLATILITY_CSV,
    VOLATILITY_RESULTS_DIR,
)
from multi_scale_volatility.research.decomposition import DecompositionPaths, run_decomposition
from multi_scale_volatility.research.global_diagnosis.baselines import BaselinePaths, create_baselines
from multi_scale_volatility.research.global_diagnosis.entropy import (
    DELAY,
    EMBEDDING_DIMENSION,
    JITTER_MAGNITUDE,
    JITTER_SEED,
    EntropyPaths,
    compute_entropy_metrics,
)
from multi_scale_volatility.research.global_diagnosis.monte_carlo_metrics import (
    MonteCarloMetricPaths,
    compute_monte_carlo_comparisons,
    compute_monte_carlo_metrics,
)
from multi_scale_volatility.research.global_diagnosis.volatility import (
    VolatilityPaths,
    compute_volatility_metrics,
)
from multi_scale_volatility.research.length_standardization import (
    LengthStandardizationPaths,
    standardize_length,
)
from multi_scale_volatility.research.preprocessing import PreprocessingPaths, run_preprocessing


def add_global_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    _add_preprocess(subparsers)
    _add_standardize(subparsers)
    _add_baselines(subparsers)
    _add_decompose(subparsers)
    _add_volatility(subparsers)
    _add_entropy(subparsers)
    _add_monte_carlo_metrics(subparsers)
    _add_monte_carlo_comparisons(subparsers)


def _add_preprocess(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("preprocess", help="Preprocess raw MetaTrader CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_METATRADER_DIR)
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.set_defaults(handler=_handle_preprocess)


def _add_standardize(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "standardize",
        aliases=["length-standardization"],
        help="Trim clean returns to a dyadic length.",
    )
    parser.add_argument("--input-csv", type=Path, default=CLEAN_RETURNS_CSV)
    parser.add_argument("--output-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--report-json", type=Path, default=TRUNCATION_REPORT_JSON)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_standardize)


def _add_baselines(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "baselines", help="Create Monte Carlo baseline series and decompositions."
    )
    parser.add_argument("--input-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--data-dir", type=Path, default=MONTE_CARLO_BASELINES_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_baselines)


def _add_decompose(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("decompose", help="Create dyadic decompositions.")
    _add_decomposition_inputs(parser)
    parser.add_argument("--output-dir", type=Path, default=DECOMPOSITION_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_decompose)


def _add_volatility(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("volatility", help="Compute volatility metrics.")
    _add_decomposition_metric_inputs(parser)
    parser.add_argument("--output-dir", type=Path, default=VOLATILITY_RESULTS_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_volatility)


def _add_entropy(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("entropy", help="Compute entropy metrics.")
    _add_decomposition_metric_inputs(parser)
    parser.add_argument("--output-dir", type=Path, default=ENTROPY_RESULTS_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--embedding-dimension", type=int, default=EMBEDDING_DIMENSION)
    parser.add_argument("--delay", type=int, default=DELAY)
    parser.add_argument("--jitter-seed", type=int, default=JITTER_SEED)
    parser.add_argument("--jitter-magnitude", type=float, default=JITTER_MAGNITUDE)
    parser.set_defaults(handler=_handle_entropy)


def _add_monte_carlo_metrics(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "monte-carlo-metrics",
        help="Compute metric tables and summaries for Monte Carlo baselines.",
    )
    parser.add_argument("--audit-csv", type=Path, default=MONTE_CARLO_BASELINE_AUDIT_CSV)
    parser.add_argument("--results-dir", type=Path, default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    parser.add_argument(
        "--runtime-log-csv",
        type=Path,
        default=MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV,
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_monte_carlo_metrics)


def _add_monte_carlo_comparisons(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "monte-carlo-comparisons",
        help="Create empirical-vs-envelope comparison tables from existing MC metrics.",
    )
    parser.add_argument("--results-dir", type=Path, default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    parser.add_argument("--final-returns-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--final-decomposition-csv", type=Path, default=FINAL_DECOMPOSITION_CSV)
    parser.add_argument("--empirical-volatility-csv", type=Path, default=VOLATILITY_CSV)
    parser.add_argument("--empirical-entropy-csv", type=Path, default=LAYER_ENTROPY_CSV)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_monte_carlo_comparisons)


def _add_decomposition_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--shuffle-csv", type=Path, default=SHUFFLE_RETURNS_CSV)
    parser.add_argument("--gaussian-csv", type=Path, default=GAUSSIAN_RETURNS_CSV)


def _add_decomposition_metric_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-decomposition-csv", type=Path, default=FINAL_DECOMPOSITION_CSV)
    parser.add_argument(
        "--shuffle-decomposition-csv",
        type=Path,
        default=SHUFFLE_DECOMPOSITION_CSV,
    )
    parser.add_argument(
        "--gaussian-decomposition-csv",
        type=Path,
        default=GAUSSIAN_DECOMPOSITION_CSV,
    )


def _handle_preprocess(args: argparse.Namespace) -> None:
    report = run_preprocessing(
        PreprocessingPaths(raw_dir=args.raw_dir, intermediate_dir=args.intermediate_dir)
    )
    print_json(report["outputs"])


def _handle_standardize(args: argparse.Namespace) -> None:
    report = standardize_length(
        LengthStandardizationPaths(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            report_json=args.report_json,
        ),
        k=args.k,
    )
    print_json(report)


def _handle_baselines(args: argparse.Namespace) -> None:
    report = create_baselines(
        BaselinePaths(
            input_csv=args.input_csv,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
        ),
        k=args.k,
    )
    print_json(report)


def _handle_decompose(args: argparse.Namespace) -> None:
    report = run_decomposition(
        DecompositionPaths(
            final_csv=args.final_csv,
            shuffle_csv=args.shuffle_csv,
            gaussian_csv=args.gaussian_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
    )
    print_json(report)


def _handle_volatility(args: argparse.Namespace) -> None:
    report = compute_volatility_metrics(
        VolatilityPaths(
            final_decomposition_csv=args.final_decomposition_csv,
            shuffle_decomposition_csv=args.shuffle_decomposition_csv,
            gaussian_decomposition_csv=args.gaussian_decomposition_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
    )
    print_json(report)


def _handle_entropy(args: argparse.Namespace) -> None:
    report = compute_entropy_metrics(
        EntropyPaths(
            final_decomposition_csv=args.final_decomposition_csv,
            shuffle_decomposition_csv=args.shuffle_decomposition_csv,
            gaussian_decomposition_csv=args.gaussian_decomposition_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
        embedding_dimension=args.embedding_dimension,
        delay=args.delay,
        jitter_seed=args.jitter_seed,
        jitter_magnitude=args.jitter_magnitude,
    )
    print_json(report)


def _handle_monte_carlo_metrics(args: argparse.Namespace) -> None:
    report = compute_monte_carlo_metrics(
        MonteCarloMetricPaths(
            audit_csv=args.audit_csv,
            results_dir=args.results_dir,
            runtime_log_csv=args.runtime_log_csv,
        ),
        k=args.k,
    )
    print_json(report)


def _handle_monte_carlo_comparisons(args: argparse.Namespace) -> None:
    report = compute_monte_carlo_comparisons(
        MonteCarloMetricPaths(
            results_dir=args.results_dir,
            final_returns_csv=args.final_returns_csv,
            final_decomposition_csv=args.final_decomposition_csv,
            empirical_volatility_csv=args.empirical_volatility_csv,
            empirical_entropy_csv=args.empirical_entropy_csv,
        ),
        k=args.k,
    )
    print_json(report)
