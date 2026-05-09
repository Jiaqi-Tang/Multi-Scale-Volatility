from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import PreprocessingPaths, run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw HistData MetaTrader EUR/USD M1 CSVs."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/metatrader"))
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preprocessing(
        PreprocessingPaths(
            raw_dir=args.raw_dir,
            intermediate_dir=args.intermediate_dir,
        )
    )
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    main()
