from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import BaselinePaths, GAUSSIAN_SEED, SHUFFLE_SEED, create_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shuffled and Gaussian baseline return series."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/final/eurusd_5m_log_returns_final.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/baselines"),
    )
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--gaussian-seed", type=int, default=GAUSSIAN_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = create_baselines(
        BaselinePaths(input_csv=args.input_csv, output_dir=args.output_dir),
        shuffle_seed=args.shuffle_seed,
        gaussian_seed=args.gaussian_seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
