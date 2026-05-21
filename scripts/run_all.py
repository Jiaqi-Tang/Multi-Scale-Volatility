from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "run_preprocessing.py",
    "run_length_standardization.py",
    "run_baselines.py",
    "run_decomposition.py",
    "run_volatility.py",
    "run_entropy.py",
    "run_eda_plots.py",
    "run_decomposition_plots.py",
    "run_volatility_plots.py",
    "run_entropy_plots.py",
    "run_memo_plots.py",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"

    for script_name in SCRIPT_ORDER:
        script_path = scripts_dir / script_name
        print(f"Running {script_path.relative_to(repo_root)}")
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            check=True,
        )


if __name__ == "__main__":
    main()
