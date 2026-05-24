"""Command-line interface for the volatility entropy pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from src.baselines import BaselinePaths, create_baselines
from src.decomposition import DecompositionPaths, run_decomposition
from src.entropy import (
    DELAY,
    EMBEDDING_DIMENSION,
    JITTER_MAGNITUDE,
    JITTER_SEED,
    EntropyPaths,
    compute_entropy_metrics,
)
from src.config.constants import DEFAULT_K, GAUSSIAN_SEED, SHUFFLE_SEED
from src.config.paths import (
    BASELINES_DIR,
    CLEAN_RETURNS_CSV,
    DECOMPOSITION_DIR,
    DECOMPOSITION_PLOTS_DIR,
    EDA_PLOTS_DIR,
    ENTROPY_GAPS_CSV,
    ENTROPY_PLOTS_DIR,
    ENTROPY_REPORT_JSON,
    ENTROPY_RESULTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    GAUSSIAN_RETURNS_CSV,
    INTERMEDIATE_DIR,
    LAYER_ENTROPY_CSV,
    MEMO_PLOTS_DIR,
    RAW_METATRADER_DIR,
    SHUFFLE_DECOMPOSITION_CSV,
    SHUFFLE_RETURNS_CSV,
    TRUNCATION_REPORT_JSON,
    VOLATILITY_CSV,
    VOLATILITY_PLOTS_DIR,
    VOLATILITY_RESULTS_DIR,
)
from src.length_standardization import LengthStandardizationPaths, standardize_length
from src.pipeline import PipelineOptions, run_all, run_core_pipeline, run_plot_pipeline
from src.plotting.decomposition import DecompositionPlotPaths, create_decomposition_plots
from src.plotting.eda import PlotPaths, create_eda_plots
from src.plotting.entropy import EntropyPlotPaths, create_entropy_plots
from src.plotting.memo import MemoPlotPaths, create_memo_plots
from src.plotting.volatility import VolatilityPlotPaths, create_volatility_plots
from src.preprocessing import PreprocessingPaths, run_preprocessing
from src.volatility import VolatilityPaths, compute_volatility_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
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
    _add_plot(subparsers)
    _add_run_all(subparsers)

    return parser


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
    parser = subparsers.add_parser("baselines", help="Create baseline return series.")
    parser.add_argument("--input-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--output-dir", type=Path, default=BASELINES_DIR)
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--gaussian-seed", type=int, default=GAUSSIAN_SEED)
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


def _add_plot(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("plot", help="Create plot artifacts.")
    plot_subparsers = parser.add_subparsers(dest="plot_command")

    eda = plot_subparsers.add_parser("eda", help="Create return EDA plots.")
    _add_return_inputs(eda)
    eda.add_argument("--output-dir", type=Path, default=EDA_PLOTS_DIR)
    eda.add_argument("--max-acf-lag", type=int, default=288)
    eda.set_defaults(handler=_handle_plot_eda)

    decomposition = plot_subparsers.add_parser(
        "decomposition",
        help="Create decomposition plots.",
    )
    _add_decomposition_csv_inputs(decomposition)
    decomposition.add_argument("--output-dir", type=Path, default=DECOMPOSITION_PLOTS_DIR)
    decomposition.add_argument("--k", type=int, default=DEFAULT_K)
    decomposition.add_argument("--short-max-acf-lag", type=int, default=1440)
    decomposition.add_argument("--long-max-acf-lag", type=int, default=6336)
    decomposition.set_defaults(handler=_handle_plot_decomposition)

    volatility = plot_subparsers.add_parser("volatility", help="Create volatility plots.")
    volatility.add_argument("--volatility-csv", type=Path, default=VOLATILITY_CSV)
    volatility.add_argument("--output-dir", type=Path, default=VOLATILITY_PLOTS_DIR)
    volatility.add_argument("--k", type=int, default=DEFAULT_K)
    volatility.set_defaults(handler=_handle_plot_volatility)

    entropy = plot_subparsers.add_parser("entropy", help="Create entropy plots.")
    entropy.add_argument("--layer-entropy-csv", type=Path, default=LAYER_ENTROPY_CSV)
    entropy.add_argument("--entropy-gaps-csv", type=Path, default=ENTROPY_GAPS_CSV)
    entropy.add_argument("--entropy-report-json", type=Path, default=ENTROPY_REPORT_JSON)
    entropy.add_argument("--output-dir", type=Path, default=ENTROPY_PLOTS_DIR)
    entropy.add_argument("--k", type=int, default=DEFAULT_K)
    entropy.set_defaults(handler=_handle_plot_entropy)

    memo = plot_subparsers.add_parser("memo", help="Create memo figures.")
    _add_return_inputs(memo)
    memo.add_argument("--final-decomposition-csv", type=Path, default=FINAL_DECOMPOSITION_CSV)
    memo.add_argument(
        "--shuffle-decomposition-csv",
        type=Path,
        default=SHUFFLE_DECOMPOSITION_CSV,
    )
    memo.add_argument("--volatility-csv", type=Path, default=VOLATILITY_CSV)
    memo.add_argument("--layer-entropy-csv", type=Path, default=LAYER_ENTROPY_CSV)
    memo.add_argument("--output-dir", type=Path, default=MEMO_PLOTS_DIR)
    memo.add_argument("--k", type=int, default=DEFAULT_K)
    memo.set_defaults(handler=_handle_plot_memo)

    all_plots = plot_subparsers.add_parser("all", help="Create all plot artifacts.")
    all_plots.add_argument("--k", type=int, default=DEFAULT_K)
    all_plots.set_defaults(handler=_handle_plot_all)


def _add_run_all(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run-all", help="Run the full in-process pipeline.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--skip-plots", action="store_true")
    parser.set_defaults(handler=_handle_run_all)


def _add_return_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--shuffle-csv", type=Path, default=SHUFFLE_RETURNS_CSV)
    parser.add_argument("--gaussian-csv", type=Path, default=GAUSSIAN_RETURNS_CSV)


def _add_decomposition_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-csv", type=Path, default=FINAL_RETURNS_CSV)
    parser.add_argument("--shuffle-csv", type=Path, default=SHUFFLE_RETURNS_CSV)
    parser.add_argument("--gaussian-csv", type=Path, default=GAUSSIAN_RETURNS_CSV)


def _add_decomposition_csv_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final-csv", type=Path, default=FINAL_DECOMPOSITION_CSV)
    parser.add_argument("--shuffle-csv", type=Path, default=SHUFFLE_DECOMPOSITION_CSV)
    parser.add_argument("--gaussian-csv", type=Path, default=GAUSSIAN_DECOMPOSITION_CSV)


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
        BaselinePaths(input_csv=args.input_csv, output_dir=args.output_dir),
        shuffle_seed=args.shuffle_seed,
        gaussian_seed=args.gaussian_seed,
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


def _handle_plot_eda(args: argparse.Namespace) -> None:
    _print_paths(
        create_eda_plots(
            PlotPaths(
                final_csv=args.final_csv,
                shuffle_csv=args.shuffle_csv,
                gaussian_csv=args.gaussian_csv,
                output_dir=args.output_dir,
            ),
            max_acf_lag=args.max_acf_lag,
        )
    )


def _handle_plot_decomposition(args: argparse.Namespace) -> None:
    _print_paths(
        create_decomposition_plots(
            DecompositionPlotPaths(
                final_csv=args.final_csv,
                shuffle_csv=args.shuffle_csv,
                gaussian_csv=args.gaussian_csv,
                output_dir=args.output_dir,
            ),
            k=args.k,
            short_max_acf_lag=args.short_max_acf_lag,
            long_max_acf_lag=args.long_max_acf_lag,
        )
    )


def _handle_plot_volatility(args: argparse.Namespace) -> None:
    _print_paths(
        create_volatility_plots(
            VolatilityPlotPaths(
                volatility_csv=args.volatility_csv,
                output_dir=args.output_dir,
            ),
            k=args.k,
        )
    )


def _handle_plot_entropy(args: argparse.Namespace) -> None:
    _print_paths(
        create_entropy_plots(
            EntropyPlotPaths(
                layer_entropy_csv=args.layer_entropy_csv,
                entropy_gaps_csv=args.entropy_gaps_csv,
                entropy_report_json=args.entropy_report_json,
                output_dir=args.output_dir,
            ),
            k=args.k,
        )
    )


def _handle_plot_memo(args: argparse.Namespace) -> None:
    _print_paths(
        create_memo_plots(
            MemoPlotPaths(
                final_returns_csv=args.final_csv,
                shuffle_returns_csv=args.shuffle_csv,
                gaussian_returns_csv=args.gaussian_csv,
                final_decomposition_csv=args.final_decomposition_csv,
                shuffle_decomposition_csv=args.shuffle_decomposition_csv,
                volatility_csv=args.volatility_csv,
                layer_entropy_csv=args.layer_entropy_csv,
                output_dir=args.output_dir,
            ),
            k=args.k,
        )
    )


def _handle_plot_all(args: argparse.Namespace) -> None:
    results = run_plot_pipeline(PipelineOptions(k=args.k))
    for outputs in results.values():
        _print_paths(outputs)


def _handle_run_all(args: argparse.Namespace) -> None:
    results = run_all(PipelineOptions(k=args.k, include_plots=not args.skip_plots))
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
