from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.length_standardization import (
    DEFAULT_K,
    LengthStandardizationPaths,
    standardize_length,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim clean EUR/USD 5m returns to a length divisible by 2**K."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/intermediate/eurusd_5m_log_returns_clean.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/final/eurusd_5m_log_returns_final.csv"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("data/final/truncation_report.json"),
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = standardize_length(
        LengthStandardizationPaths(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            report_json=args.report_json,
        ),
        k=args.k,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
