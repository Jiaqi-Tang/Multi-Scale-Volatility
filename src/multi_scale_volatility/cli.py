"""Command-line interface for the volatility entropy pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from multi_scale_volatility.baselines import BaselinePaths, create_baselines
from multi_scale_volatility.decomposition import DecompositionPaths, run_decomposition
from multi_scale_volatility.entropy import (
    DELAY,
    EMBEDDING_DIMENSION,
    JITTER_MAGNITUDE,
    JITTER_SEED,
    EntropyPaths,
    compute_entropy_metrics,
)
from multi_scale_volatility.config.names import DEFAULT_K
from multi_scale_volatility.config.paths import (
    CLEAN_RETURNS_CSV,
    DECOMPOSITION_DIR,
    ENTROPY_RESULTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    GAUSSIAN_RETURNS_CSV,
    INTERMEDIATE_DIR,
    LAYER_ENTROPY_CSV,
    MEMO_PLOTS_DIR,
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV,
    MONTE_CARLO_BASELINES_DATA_DIR,
    MONTE_CARLO_BASELINES_RESULTS_DIR,
    RAW_METATRADER_DIR,
    ROLLING_PLOTS_DIR,
    ROLLING_RESULTS_DIR,
    SHUFFLE_DECOMPOSITION_CSV,
    SHUFFLE_RETURNS_CSV,
    TRUNCATION_REPORT_JSON,
    VOLATILITY_CSV,
    VOLATILITY_RESULTS_DIR,
)
from multi_scale_volatility.length_standardization import LengthStandardizationPaths, standardize_length
from multi_scale_volatility.monte_carlo_metrics import (
    MonteCarloMetricPaths,
    compute_monte_carlo_comparisons,
    compute_monte_carlo_metrics,
)
from multi_scale_volatility.pipeline import PipelineOptions, run_all, run_core_pipeline, run_plot_pipeline
from multi_scale_volatility.plotting.monte_carlo_baselines import (
    MonteCarloBaselinePlotPaths,
    create_monte_carlo_baseline_plots,
    create_v11_memo_plots,
)
from multi_scale_volatility.plotting.rolling import (
    RollingExamplePlotPaths,
    RollingPlotPaths,
    create_rolling_plots,
    create_rolling_example_decomposition_plots,
)
from multi_scale_volatility.preprocessing import PreprocessingPaths, run_preprocessing
from multi_scale_volatility.runtime import configure_logging
from multi_scale_volatility.rolling import (
    ROLLING_K,
    ROLLING_STEP_SIZE,
    ROLLING_WINDOW_LENGTHS,
    RollingPaths,
    compute_rolling_decomposition_diagnostics,
)
from multi_scale_volatility.volatility import VolatilityPaths, compute_volatility_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    handler(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ve",
        description="Run EUR/USD volatility entropy pipeline stages.",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_preprocess(subparsers)
    _add_standardize(subparsers)
    _add_baselines(subparsers)
    _add_decompose(subparsers)
    _add_volatility(subparsers)
    _add_entropy(subparsers)
    _add_monte_carlo_metrics(subparsers)
    _add_monte_carlo_comparisons(subparsers)
    _add_rolling(subparsers)
    _add_plot(subparsers)
    _add_run_all(subparsers)

    return parser


def _add_preprocess(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "preprocess", help="Preprocess raw MetaTrader CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_METATRADER_DIR)
    parser.add_argument("--intermediate-dir", type=Path,
                        default=INTERMEDIATE_DIR)
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
    parser.add_argument("--report-json", type=Path,
                        default=TRUNCATION_REPORT_JSON)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_standardize)


def _add_baselines(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "baselines", help="Create Monte Carlo baseline series and decompositions.")
    parser.add_argument("--input-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--data-dir", type=Path,
                        default=MONTE_CARLO_BASELINES_DATA_DIR)
    parser.add_argument("--results-dir", type=Path,
                        default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_baselines)


def _add_decompose(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "decompose", help="Create dyadic decompositions.")
    _add_decomposition_inputs(parser)
    parser.add_argument("--output-dir", type=Path, default=DECOMPOSITION_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_decompose)


def _add_volatility(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "volatility", help="Compute volatility metrics.")
    _add_decomposition_metric_inputs(parser)
    parser.add_argument("--output-dir", type=Path,
                        default=VOLATILITY_RESULTS_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_volatility)


def _add_entropy(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("entropy", help="Compute entropy metrics.")
    _add_decomposition_metric_inputs(parser)
    parser.add_argument("--output-dir", type=Path, default=ENTROPY_RESULTS_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--embedding-dimension", type=int,
                        default=EMBEDDING_DIMENSION)
    parser.add_argument("--delay", type=int, default=DELAY)
    parser.add_argument("--jitter-seed", type=int, default=JITTER_SEED)
    parser.add_argument("--jitter-magnitude", type=float,
                        default=JITTER_MAGNITUDE)
    parser.set_defaults(handler=_handle_entropy)


def _add_monte_carlo_metrics(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "monte-carlo-metrics",
        help="Compute metric tables and summaries for Monte Carlo baselines.",
    )
    parser.add_argument("--audit-csv", type=Path,
                        default=MONTE_CARLO_BASELINE_AUDIT_CSV)
    parser.add_argument("--results-dir", type=Path,
                        default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    parser.add_argument("--runtime-log-csv", type=Path,
                        default=MONTE_CARLO_BASELINE_RUNTIME_LOG_CSV)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_monte_carlo_metrics)


def _add_monte_carlo_comparisons(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "monte-carlo-comparisons",
        help="Create empirical-vs-envelope comparison tables from existing MC metrics.",
    )
    parser.add_argument("--results-dir", type=Path,
                        default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    parser.add_argument("--final-returns-csv", type=Path,
                        default=FINAL_RETURNS_CSV)
    parser.add_argument("--final-decomposition-csv", type=Path,
                        default=FINAL_DECOMPOSITION_CSV)
    parser.add_argument("--empirical-volatility-csv", type=Path,
                        default=VOLATILITY_CSV)
    parser.add_argument("--empirical-entropy-csv", type=Path,
                        default=LAYER_ENTROPY_CSV)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.set_defaults(handler=_handle_monte_carlo_comparisons)


def _add_rolling(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "rolling",
        help="Create V2.1 rolling window decomposition diagnostics.",
    )
    parser.add_argument("--input-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--output-dir", type=Path, default=ROLLING_RESULTS_DIR)
    parser.add_argument("--window-lengths", type=int, nargs="+", default=list(ROLLING_WINDOW_LENGTHS))
    parser.add_argument("--step-size", type=int, default=ROLLING_STEP_SIZE)
    parser.add_argument("--k", type=int, default=ROLLING_K)
    parser.set_defaults(handler=_handle_rolling)


def _add_plot(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("plot", help="Create plot artifacts.")
    plot_subparsers = parser.add_subparsers(dest="plot_command")

    memo = plot_subparsers.add_parser("memo", help="Create memo figures.")
    memo.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    memo.add_argument("--final-decomposition-csv", type=Path,
                      default=FINAL_DECOMPOSITION_CSV)
    memo.add_argument("--results-dir", type=Path,
                      default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    memo.add_argument("--audit-csv", type=Path,
                      default=MONTE_CARLO_BASELINE_AUDIT_CSV)
    memo.add_argument("--output-dir", type=Path, default=MEMO_PLOTS_DIR)
    memo.add_argument("--k", type=int, default=DEFAULT_K)
    memo.set_defaults(handler=_handle_plot_memo)

    mc_baselines = plot_subparsers.add_parser(
        "monte-carlo-baselines",
        help="Create V1.1 Monte Carlo baseline envelope plots.",
    )
    mc_baselines.add_argument("--results-dir", type=Path,
                              default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    mc_baselines.add_argument("--final-returns-csv", type=Path,
                              default=FINAL_RETURNS_CSV)
    mc_baselines.add_argument("--final-decomposition-csv", type=Path,
                              default=FINAL_DECOMPOSITION_CSV)
    mc_baselines.add_argument("--k", type=int, default=DEFAULT_K)
    mc_baselines.set_defaults(handler=_handle_plot_monte_carlo_baselines)

    rolling = plot_subparsers.add_parser(
        "rolling",
        help="Create V2.1 rolling volatility and scale-composition plots.",
    )
    rolling.add_argument("--results-dir", type=Path, default=ROLLING_RESULTS_DIR)
    rolling.add_argument("--output-dir", type=Path, default=ROLLING_PLOTS_DIR)
    rolling.set_defaults(handler=_handle_plot_rolling)

    rolling_examples = plot_subparsers.add_parser(
        "rolling-examples",
        help="Create V2.1 example rolling decomposition plots.",
    )
    rolling_examples.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    rolling_examples.add_argument("--results-dir", type=Path, default=ROLLING_RESULTS_DIR)
    rolling_examples.add_argument(
        "--output-dir",
        type=Path,
        default=ROLLING_PLOTS_DIR / "examples",
    )
    rolling_examples.add_argument("--k", type=int, default=ROLLING_K)
    rolling_examples.add_argument("--step-size", type=int, default=ROLLING_STEP_SIZE)
    rolling_examples.set_defaults(handler=_handle_plot_rolling_examples)

    all_plots = plot_subparsers.add_parser(
        "all", help="Create all plot artifacts.")
    all_plots.add_argument("--k", type=int, default=DEFAULT_K)
    all_plots.set_defaults(handler=_handle_plot_all)


def _add_run_all(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run-all", help="Run the full in-process pipeline.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--skip-plots", action="store_true")
    parser.set_defaults(handler=_handle_run_all)


def _add_decomposition_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--shuffle-csv", type=Path,
                        default=SHUFFLE_RETURNS_CSV)
    parser.add_argument("--gaussian-csv", type=Path,
                        default=GAUSSIAN_RETURNS_CSV)


def _add_decomposition_metric_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--final-decomposition-csv",
        type=Path,
        default=FINAL_DECOMPOSITION_CSV,
    )
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
        PreprocessingPaths(
            raw_dir=args.raw_dir,
            intermediate_dir=args.intermediate_dir,
        )
    )
    _print_json(report["outputs"])


def _handle_standardize(args: argparse.Namespace) -> None:
    report = standardize_length(
        LengthStandardizationPaths(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            report_json=args.report_json,
        ),
        k=args.k,
    )
    _print_json(report)


def _handle_baselines(args: argparse.Namespace) -> None:
    report = create_baselines(
        BaselinePaths(
            input_csv=args.input_csv,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
        ),
        k=args.k,
    )
    _print_json(report)


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
    _print_json(report)


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
    _print_json(report)


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
    _print_json(report)


def _handle_monte_carlo_metrics(args: argparse.Namespace) -> None:
    report = compute_monte_carlo_metrics(
        MonteCarloMetricPaths(
            audit_csv=args.audit_csv,
            results_dir=args.results_dir,
            runtime_log_csv=args.runtime_log_csv,
        ),
        k=args.k,
    )
    _print_json(report)


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
    _print_json(report)


def _handle_rolling(args: argparse.Namespace) -> None:
    report = compute_rolling_decomposition_diagnostics(
        RollingPaths(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
        ),
        window_lengths=tuple(args.window_lengths),
        step_size=args.step_size,
        k=args.k,
    )
    _print_json(report)


def _handle_plot_memo(args: argparse.Namespace) -> None:
    _print_paths(
        create_v11_memo_plots(
            MonteCarloBaselinePlotPaths(
                results_dir=args.results_dir,
                audit_csv=args.audit_csv,
                final_returns_csv=args.final_csv,
                final_decomposition_csv=args.final_decomposition_csv,
                memo_output_dir=args.output_dir,
            ),
            k=args.k,
        )
    )


def _handle_plot_monte_carlo_baselines(args: argparse.Namespace) -> None:
    _print_paths(
        create_monte_carlo_baseline_plots(
            MonteCarloBaselinePlotPaths(
                results_dir=args.results_dir,
                final_returns_csv=args.final_returns_csv,
                final_decomposition_csv=args.final_decomposition_csv,
            ),
            k=args.k,
        )
    )


def _handle_plot_rolling(args: argparse.Namespace) -> None:
    _print_paths(
        create_rolling_plots(
            RollingPlotPaths(
                results_dir=args.results_dir,
                output_dir=args.output_dir,
            )
        )
    )


def _handle_plot_rolling_examples(args: argparse.Namespace) -> None:
    _print_paths(
        create_rolling_example_decomposition_plots(
            RollingExamplePlotPaths(
                final_returns_csv=args.final_csv,
                results_dir=args.results_dir,
                output_dir=args.output_dir,
            ),
            k=args.k,
            step_size=args.step_size,
        )
    )


def _handle_plot_all(args: argparse.Namespace) -> None:
    results = run_plot_pipeline(PipelineOptions(k=args.k))
    for outputs in results.values():
        _print_paths(outputs)


def _handle_run_all(args: argparse.Namespace) -> None:
    results = run_all(PipelineOptions(
        k=args.k, include_plots=not args.skip_plots))
    _print_json(_json_ready_summary(results))


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _print_paths(paths: list[Path]) -> None:
    for path in paths:
        print(path)


def _json_ready_summary(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, list):
            summary[key] = [str(item) for item in value]
        else:
            summary[key] = value
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
