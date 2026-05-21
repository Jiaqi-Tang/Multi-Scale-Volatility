from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.globals.paths import (
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    GAUSSIAN_RETURNS_CSV,
    LAYER_ENTROPY_CSV,
    MEMO_PLOTS_DIR,
    SHUFFLE_DECOMPOSITION_CSV,
    SHUFFLE_RETURNS_CSV,
    VOLATILITY_CSV,
)
from src.plotting.memo import MemoPlotPaths, create_memo_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create polished figures for Memo.md.")
    parser.add_argument(
        "--final-returns-csv",
        type=Path,
        default=FINAL_RETURNS_CSV,
    )
    parser.add_argument(
        "--shuffle-returns-csv",
        type=Path,
        default=SHUFFLE_RETURNS_CSV,
    )
    parser.add_argument(
        "--gaussian-returns-csv",
        type=Path,
        default=GAUSSIAN_RETURNS_CSV,
    )
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
    parser.add_argument("--volatility-csv", type=Path, default=VOLATILITY_CSV)
    parser.add_argument("--layer-entropy-csv", type=Path, default=LAYER_ENTROPY_CSV)
    parser.add_argument("--output-dir", type=Path, default=MEMO_PLOTS_DIR)
    parser.add_argument("--k", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_memo_plots(
        MemoPlotPaths(
            final_returns_csv=args.final_returns_csv,
            shuffle_returns_csv=args.shuffle_returns_csv,
            gaussian_returns_csv=args.gaussian_returns_csv,
            final_decomposition_csv=args.final_decomposition_csv,
            shuffle_decomposition_csv=args.shuffle_decomposition_csv,
            volatility_csv=args.volatility_csv,
            layer_entropy_csv=args.layer_entropy_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
