"""Plotting command-line commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from multi_scale_volatility.app.cli_common import print_paths
from multi_scale_volatility.app.pipeline import PipelineOptions, run_plot_pipeline
from multi_scale_volatility.core.config.names import DEFAULT_K
from multi_scale_volatility.core.config.paths import (
    EVENT_EDA_PLOTS_DIR,
    EVENT_STUDY_RESULTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    MEMO_PLOTS_DIR,
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    MONTE_CARLO_BASELINES_RESULTS_DIR,
    ROLLING_BASELINE_PLOTS_DIR,
    ROLLING_BASELINE_RESULTS_DIR,
    ROLLING_REGIME_PLOTS_DIR,
    ROLLING_REGIME_RESULTS_DIR,
    ROLLING_RESULTS_DIR,
    ROLLING_WINDOWS_PLOTS_DIR,
)
from multi_scale_volatility.plotting.event_eda import EventEdaPlotPaths, create_event_eda_plots
from multi_scale_volatility.plotting.global_results import (
    MonteCarloBaselinePlotPaths,
    create_monte_carlo_baseline_plots,
    create_v2_memo_plots,
)
from multi_scale_volatility.plotting.rolling import (
    RollingExamplePlotPaths,
    RollingPlotPaths,
    create_rolling_example_decomposition_plots,
    create_rolling_plots,
)
from multi_scale_volatility.plotting.rolling_baselines import (
    RollingBaselinePlotPaths,
    create_rolling_baseline_plots,
)
from multi_scale_volatility.plotting.rolling_regimes import (
    RollingRegimePlotPaths,
    create_rolling_regime_plots,
)
from multi_scale_volatility.research.rolling_window_diagnosis.rolling import (
    ROLLING_K,
    ROLLING_STEP_SIZE,
)


def add_plot_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("plot", help="Create plot artifacts.")
    plot_subparsers = parser.add_subparsers(dest="plot_command")

    memo = plot_subparsers.add_parser("memo", help="Create memo figures.")
    memo.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    memo.add_argument("--final-decomposition-csv", type=Path, default=FINAL_DECOMPOSITION_CSV)
    memo.add_argument("--results-dir", type=Path, default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    memo.add_argument("--audit-csv", type=Path, default=MONTE_CARLO_BASELINE_AUDIT_CSV)
    memo.add_argument("--output-dir", type=Path, default=MEMO_PLOTS_DIR)
    memo.add_argument("--k", type=int, default=DEFAULT_K)
    memo.set_defaults(handler=_handle_plot_memo)

    mc_baselines = plot_subparsers.add_parser(
        "monte-carlo-baselines",
        help="Create global Monte Carlo baseline envelope plots.",
    )
    mc_baselines.add_argument("--results-dir", type=Path, default=MONTE_CARLO_BASELINES_RESULTS_DIR)
    mc_baselines.add_argument("--final-returns-csv", type=Path, default=FINAL_RETURNS_CSV)
    mc_baselines.add_argument(
        "--final-decomposition-csv",
        type=Path,
        default=FINAL_DECOMPOSITION_CSV,
    )
    mc_baselines.add_argument("--k", type=int, default=DEFAULT_K)
    mc_baselines.set_defaults(handler=_handle_plot_monte_carlo_baselines)

    rolling = plot_subparsers.add_parser(
        "rolling",
        help="Create rolling volatility and scale-composition plots.",
    )
    rolling.add_argument("--results-dir", type=Path, default=ROLLING_RESULTS_DIR)
    rolling.add_argument("--output-dir", type=Path, default=ROLLING_WINDOWS_PLOTS_DIR)
    rolling.set_defaults(handler=_handle_plot_rolling)

    rolling_examples = plot_subparsers.add_parser(
        "rolling-examples",
        help="Create example rolling decomposition plots.",
    )
    rolling_examples.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    rolling_examples.add_argument("--results-dir", type=Path, default=ROLLING_RESULTS_DIR)
    rolling_examples.add_argument("--output-dir", type=Path, default=ROLLING_WINDOWS_PLOTS_DIR / "examples")
    rolling_examples.add_argument("--k", type=int, default=ROLLING_K)
    rolling_examples.add_argument("--step-size", type=int, default=ROLLING_STEP_SIZE)
    rolling_examples.set_defaults(handler=_handle_plot_rolling_examples)

    rolling_baselines = plot_subparsers.add_parser(
        "rolling-baselines",
        help="Create rolling baseline correlation envelope plots.",
    )
    rolling_baselines.add_argument("--results-dir", type=Path, default=ROLLING_BASELINE_RESULTS_DIR)
    rolling_baselines.add_argument("--output-dir", type=Path, default=ROLLING_BASELINE_PLOTS_DIR)
    rolling_baselines.set_defaults(handler=_handle_plot_rolling_baselines)

    rolling_regimes = plot_subparsers.add_parser(
        "rolling-regimes",
        help="Create rolling volatility-state regime plots.",
    )
    rolling_regimes.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    rolling_regimes.add_argument("--results-dir", type=Path, default=ROLLING_REGIME_RESULTS_DIR)
    rolling_regimes.add_argument("--rolling-results-dir", type=Path, default=ROLLING_RESULTS_DIR)
    rolling_regimes.add_argument("--output-dir", type=Path, default=ROLLING_REGIME_PLOTS_DIR)
    rolling_regimes.set_defaults(handler=_handle_plot_rolling_regimes)

    events = plot_subparsers.add_parser(
        "events",
        help="Create V3 event-study exploratory plots.",
    )
    events.add_argument("--results-dir", type=Path, default=EVENT_STUDY_RESULTS_DIR)
    events.add_argument("--output-dir", type=Path, default=EVENT_EDA_PLOTS_DIR)
    events.add_argument("--random-seed", type=int, default=20260712)
    events.set_defaults(handler=_handle_plot_events)

    all_plots = plot_subparsers.add_parser("all", help="Create all plot artifacts.")
    all_plots.add_argument("--k", type=int, default=DEFAULT_K)
    all_plots.set_defaults(handler=_handle_plot_all)


def _handle_plot_memo(args: argparse.Namespace) -> None:
    print_paths(
        create_v2_memo_plots(
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
    print_paths(
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
    print_paths(
        create_rolling_plots(
            RollingPlotPaths(results_dir=args.results_dir, output_dir=args.output_dir)
        )
    )


def _handle_plot_rolling_examples(args: argparse.Namespace) -> None:
    print_paths(
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


def _handle_plot_rolling_baselines(args: argparse.Namespace) -> None:
    print_paths(
        create_rolling_baseline_plots(
            RollingBaselinePlotPaths(results_dir=args.results_dir, output_dir=args.output_dir)
        )
    )


def _handle_plot_rolling_regimes(args: argparse.Namespace) -> None:
    print_paths(
        create_rolling_regime_plots(
            RollingRegimePlotPaths(
                final_returns_csv=args.final_csv,
                results_dir=args.results_dir,
                rolling_results_dir=args.rolling_results_dir,
                output_dir=args.output_dir,
            )
        )
    )


def _handle_plot_events(args: argparse.Namespace) -> None:
    print_paths(
        create_event_eda_plots(
            EventEdaPlotPaths(results_dir=args.results_dir, output_dir=args.output_dir),
            random_seed=args.random_seed,
        )
    )


def _handle_plot_all(args: argparse.Namespace) -> None:
    results = run_plot_pipeline(PipelineOptions(k=args.k))
    for outputs in results.values():
        print_paths(outputs)
