"""Command-line interface for the volatility entropy pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from multi_scale_volatility.app.cli_common import json_ready_summary, print_json
from multi_scale_volatility.app.cli_global import add_global_commands
from multi_scale_volatility.app.cli_plotting import add_plot_commands
from multi_scale_volatility.app.cli_rolling import add_rolling_commands
from multi_scale_volatility.app.cli_run import add_run_commands
from multi_scale_volatility.app.pipeline import PipelineOptions, run_all
from multi_scale_volatility.app.runtime import configure_logging
from multi_scale_volatility.core.config.names import DEFAULT_K
from multi_scale_volatility.research.rolling_window_diagnosis.rolling import ROLLING_K


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

    add_global_commands(subparsers)
    add_rolling_commands(subparsers)
    add_plot_commands(subparsers)
    add_run_commands(subparsers)
    _add_run_all(subparsers)

    return parser


def _add_run_all(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run-all", help="Run the full in-process pipeline.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--rolling-k", type=int, default=ROLLING_K)
    parser.add_argument("--skip-plots", action="store_true")
    parser.set_defaults(handler=_handle_run_all)


def _handle_run_all(args: argparse.Namespace) -> None:
    results = run_all(
        PipelineOptions(
            k=args.k,
            rolling_k=args.rolling_k,
            include_plots=not args.skip_plots,
        )
    )
    print_json(json_ready_summary(results))


if __name__ == "__main__":
    raise SystemExit(main())
