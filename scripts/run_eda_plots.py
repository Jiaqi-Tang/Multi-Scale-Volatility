from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plots import PlotPaths, create_eda_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create EDA plots for EUR/USD final returns and baselines."
    )
    parser.add_argument(
        "--final-csv",
        type=Path,
        default=Path("data/final/eurusd_5m_log_returns_final.csv"),
    )
    parser.add_argument(
        "--shuffle-csv",
        type=Path,
        default=Path("data/baselines/eurusd_5m_log_returns_shuffle.csv"),
    )
    parser.add_argument(
        "--gaussian-csv",
        type=Path,
        default=Path("data/baselines/eurusd_5m_log_returns_gaussian.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots/eda"))
    parser.add_argument("--max-acf-lag", type=int, default=288)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_eda_plots(
        PlotPaths(
            final_csv=args.final_csv,
            shuffle_csv=args.shuffle_csv,
            gaussian_csv=args.gaussian_csv,
            output_dir=args.output_dir,
        ),
        max_acf_lag=args.max_acf_lag,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
