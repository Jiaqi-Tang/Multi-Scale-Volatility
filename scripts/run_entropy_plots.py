from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plots import EntropyPlotPaths, create_entropy_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create entropy result plots from layer entropy metrics."
    )
    parser.add_argument(
        "--layer-entropy-csv",
        type=Path,
        default=Path("results/entropy/layer_entropy.csv"),
    )
    parser.add_argument(
        "--entropy-gaps-csv",
        type=Path,
        default=Path("results/entropy/entropy_gaps.csv"),
    )
    parser.add_argument(
        "--entropy-report-json",
        type=Path,
        default=Path("results/entropy/entropy_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/results/entropy"),
    )
    parser.add_argument("--k", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_entropy_plots(
        EntropyPlotPaths(
            layer_entropy_csv=args.layer_entropy_csv,
            entropy_gaps_csv=args.entropy_gaps_csv,
            entropy_report_json=args.entropy_report_json,
            output_dir=args.output_dir,
        ),
        k=args.k,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
