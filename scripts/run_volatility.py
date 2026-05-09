from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.volatility import DEFAULT_K, VolatilityPaths, compute_volatility_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute volatility and energy metrics for decomposition components."
    )
    parser.add_argument(
        "--final-decomposition-csv",
        type=Path,
        default=Path("data/decomposition/final_decomposition.csv"),
    )
    parser.add_argument(
        "--shuffle-decomposition-csv",
        type=Path,
        default=Path("data/decomposition/shuffle_decomposition.csv"),
    )
    parser.add_argument(
        "--gaussian-decomposition-csv",
        type=Path,
        default=Path("data/decomposition/gaussian_decomposition.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/volatility"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compute_volatility_metrics(
        VolatilityPaths(
            final_decomposition_csv=args.final_decomposition_csv,
            shuffle_decomposition_csv=args.shuffle_decomposition_csv,
            gaussian_decomposition_csv=args.gaussian_decomposition_csv,
            output_dir=args.output_dir,
        ),
        k=args.k,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
