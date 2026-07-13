"""Nested pipeline orchestration commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from multi_scale_volatility.app.cli_common import json_ready_summary, print_json
from multi_scale_volatility.app.pipeline import (
    PipelineOptions,
    run_all,
    run_data_processing_pipeline,
    run_event_detection_stage,
    run_event_study_pipeline,
    run_event_windows_stage,
    run_event_plot_pipeline,
    run_global_analysis_pipeline,
    run_global_decompose_stage,
    run_global_metrics_stage,
    run_global_plot_pipeline,
    run_memo_plot_pipeline,
    run_monte_carlo_baselines_stage,
    run_monte_carlo_metrics_stage,
    run_monte_carlo_pipeline,
    run_plot_pipeline,
    run_preprocess_stage,
    run_regime_plot_pipeline,
    run_rolling_analysis_pipeline,
    run_rolling_baselines_stage,
    run_rolling_diagnostics_stage,
    run_rolling_plot_pipeline,
    run_rolling_regimes_stage,
    run_standardize_stage,
)
from multi_scale_volatility.app.runtime import get_logger, logged_stage
from multi_scale_volatility.core.config.names import DEFAULT_K
from multi_scale_volatility.research.rolling_window_diagnosis.rolling import ROLLING_K

logger = get_logger(__name__)


def add_run_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("run", help="Run grouped pipeline stages.")
    run_subparsers = parser.add_subparsers(dest="run_group")

    all_parser = run_subparsers.add_parser("all", help="Run the full current pipeline.")
    _add_common_options(all_parser)
    all_parser.add_argument("--skip-plots", action="store_true")
    all_parser.set_defaults(handler=_handle_run_all)

    data = run_subparsers.add_parser("data", help="Run data processing stages.")
    data.add_argument("step", nargs="?", choices=("preprocess", "standardize"))
    _add_common_options(data)
    data.set_defaults(handler=_handle_run_data)

    global_parser = run_subparsers.add_parser("global", help="Run global analysis stages.")
    global_parser.add_argument("step", nargs="?", choices=("decompose", "metrics"))
    _add_common_options(global_parser)
    global_parser.set_defaults(handler=_handle_run_global)

    monte_carlo = run_subparsers.add_parser(
        "monte-carlo",
        help="Run Monte Carlo baseline stages.",
    )
    monte_carlo.add_argument("step", nargs="?", choices=("baselines", "metrics"))
    _add_common_options(monte_carlo)
    monte_carlo.set_defaults(handler=_handle_run_monte_carlo)

    rolling = run_subparsers.add_parser("rolling", help="Run rolling analysis stages.")
    rolling.add_argument("step", nargs="?", choices=("diagnostics", "baselines", "regimes"))
    _add_common_options(rolling)
    rolling.set_defaults(handler=_handle_run_rolling)

    events = run_subparsers.add_parser("events", help="Run V3 event-study stages.")
    events.add_argument("step", nargs="?", choices=("detect", "windows"))
    events.set_defaults(handler=_handle_run_events)

    plots = run_subparsers.add_parser("plots", help="Run plot generation stages.")
    plots.add_argument("step", nargs="?", choices=("global", "rolling", "regimes", "events", "memo"))
    _add_common_options(plots)
    plots.set_defaults(handler=_handle_run_plots)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--rolling-k", type=int, default=ROLLING_K)


def _options(args: argparse.Namespace, include_plots: bool = True) -> PipelineOptions:
    return PipelineOptions(
        k=args.k,
        rolling_k=args.rolling_k,
        include_plots=include_plots,
    )


def _handle_run_all(args: argparse.Namespace) -> None:
    results = run_all(_options(args, include_plots=not args.skip_plots))
    print_json(json_ready_summary(results))


def _handle_run_data(args: argparse.Namespace) -> None:
    options = _options(args)
    if args.step is None:
        result = run_data_processing_pipeline(options)
    else:
        result = _run_named_stage(
            args.step,
            {
                "preprocess": run_preprocess_stage,
                "standardize": lambda: run_standardize_stage(options),
            },
        )
    print_json(result)


def _handle_run_global(args: argparse.Namespace) -> None:
    options = _options(args)
    if args.step is None:
        result = run_global_analysis_pipeline(options)
    else:
        result = _run_named_stage(
            args.step,
            {
                "decompose": lambda: run_global_decompose_stage(options),
                "metrics": lambda: run_global_metrics_stage(options),
            },
        )
    print_json(result)


def _handle_run_monte_carlo(args: argparse.Namespace) -> None:
    options = _options(args)
    if args.step is None:
        result = run_monte_carlo_pipeline(options)
    else:
        result = _run_named_stage(
            args.step,
            {
                "baselines": lambda: run_monte_carlo_baselines_stage(options),
                "metrics": lambda: run_monte_carlo_metrics_stage(options),
            },
        )
    print_json(result)


def _handle_run_rolling(args: argparse.Namespace) -> None:
    options = _options(args)
    if args.step is None:
        result = run_rolling_analysis_pipeline(options)
    else:
        result = _run_named_stage(
            args.step,
            {
                "diagnostics": lambda: run_rolling_diagnostics_stage(options),
                "baselines": lambda: run_rolling_baselines_stage(options),
                "regimes": run_rolling_regimes_stage,
            },
        )
    print_json(result)


def _handle_run_events(args: argparse.Namespace) -> None:
    if args.step is None:
        result = run_event_study_pipeline()
    else:
        result = _run_named_stage(
            args.step,
            {"detect": run_event_detection_stage, "windows": run_event_windows_stage},
        )
    print_json(result)


def _handle_run_plots(args: argparse.Namespace) -> None:
    options = _options(args)
    if args.step is None:
        result = run_plot_pipeline(options)
    else:
        result = _run_named_stage(
            args.step,
            {
                "global": lambda: run_global_plot_pipeline(options),
                "rolling": lambda: run_rolling_plot_pipeline(options),
                "regimes": run_regime_plot_pipeline,
                "events": run_event_plot_pipeline,
                "memo": lambda: run_memo_plot_pipeline(options),
            },
        )
    print_json(json_ready_summary(result if isinstance(result, dict) else {args.step or "plots": result}))


def _run_named_stage(
    name: str,
    stages: dict[str, Callable[[], Any]],
) -> Any:
    with logged_stage(logger, name):
        return stages[name]()
