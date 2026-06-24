"""Figure saving helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

STABLE_SAVEFIG_METADATA = {
    "CreationDate": "None",
    "Software": "multi_scale_volatility",
}


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> Path:
    fig.savefig(output_path, dpi=dpi, metadata=STABLE_SAVEFIG_METADATA)
    return output_path
