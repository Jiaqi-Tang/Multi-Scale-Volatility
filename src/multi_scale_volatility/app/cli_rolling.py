"""Rolling-window command-line commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from multi_scale_volatility.app.cli_common import print_json
from multi_scale_volatility.core.config.paths import (
    FINAL_RETURNS_CSV,
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    ROLLING_BASELINE_RESULTS_DIR,
    ROLLING_BASELINE_RUNTIME_LOG_CSV,
    ROLLING_REGIME_RESULTS_DIR,
    ROLLING_RESULTS_DIR,
)
from multi_scale_volatility.research.rolling_window_diagnosis.rolling import (
    ROLLING_K,
    ROLLING_STEP_SIZE,
    ROLLING_WINDOW_LENGTHS,
    RollingPaths,
    compute_rolling_decomposition_diagnostics,
)
from multi_scale_volatility.research.rolling_window_diagnosis.rolling_baselines import (
    RollingBaselineCorrelationPaths,
    compute_rolling_baseline_correlations,
)
from multi_scale_volatility.research.rolling_window_diagnosis.rolling_regimes import (
    MIN_EPISODE_WINDOWS,
    RollingRegimePaths,
    compute_rolling_regime_diagnostics,
)


def add_rolling_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    _add_rolling(subparsers)
    _add_rolling_baselines(subparsers)
    _add_rolling_regimes(subparsers)


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


def _add_rolling_baselines(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "rolling-baselines",
        help="Create rolling correlation envelopes for Monte Carlo baselines.",
    )
    parser.add_argument("--audit-csv", type=Path, default=MONTE_CARLO_BASELINE_AUDIT_CSV)
    parser.add_argument(
        "--empirical-layer-volatility-csv",
        type=Path,
        default=ROLLING_RESULTS_DIR / "rolling_layer_volatility.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROLLING_BASELINE_RESULTS_DIR)
    parser.add_argument("--runtime-log-csv", type=Path, default=ROLLING_BASELINE_RUNTIME_LOG_CSV)
    parser.add_argument("--window-lengths", type=int, nargs="+", default=list(ROLLING_WINDOW_LENGTHS))
    parser.add_argument("--step-size", type=int, default=ROLLING_STEP_SIZE)
    parser.add_argument("--k", type=int, default=ROLLING_K)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-simulations-per-type", type=int, default=None)
    parser.set_defaults(handler=_handle_rolling_baselines)


def _add_rolling_regimes(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "rolling-regimes",
        help="Create V2.3 rolling volatility-state regime diagnostics.",
    )
    parser.add_argument("--results-dir", type=Path, default=ROLLING_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROLLING_REGIME_RESULTS_DIR)
    parser.add_argument("--min-episode-windows", type=int, default=MIN_EPISODE_WINDOWS)
    parser.set_defaults(handler=_handle_rolling_regimes)


def _handle_rolling(args: argparse.Namespace) -> None:
    report = compute_rolling_decomposition_diagnostics(
        RollingPaths(input_csv=args.input_csv, output_dir=args.output_dir),
        window_lengths=tuple(args.window_lengths),
        step_size=args.step_size,
        k=args.k,
    )
    print_json(report)


def _handle_rolling_baselines(args: argparse.Namespace) -> None:
    report = compute_rolling_baseline_correlations(
        RollingBaselineCorrelationPaths(
            audit_csv=args.audit_csv,
            empirical_layer_volatility_csv=args.empirical_layer_volatility_csv,
            output_dir=args.output_dir,
            runtime_log_csv=args.runtime_log_csv,
        ),
        window_lengths=tuple(args.window_lengths),
        step_size=args.step_size,
        k=args.k,
        max_workers=args.max_workers,
        max_simulations_per_type=args.max_simulations_per_type,
    )
    print_json(report)


def _handle_rolling_regimes(args: argparse.Namespace) -> None:
    report = compute_rolling_regime_diagnostics(
        RollingRegimePaths(results_dir=args.results_dir, output_dir=args.output_dir),
        min_episode_windows=args.min_episode_windows,
    )
    print_json(report)
