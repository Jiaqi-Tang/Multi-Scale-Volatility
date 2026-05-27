"""Line-plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.config.columns import INDEX
from multi_scale_volatility.plotting.style import FIGURE_DPI, GRID_FIGURE_DPI
from multi_scale_volatility.scale_utils import decomposition_components


def plot_return_line(returns: np.ndarray, output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(np.arange(len(returns)), returns, linewidth=0.25, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Log return")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_decomposition_layers(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    k: int,
) -> Path:
    layers = decomposition_components(k, include_original=True)
    x = frame[INDEX].to_numpy()

    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(16, 24),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16)

    for axis, layer in zip(axes, layers):
        axis.plot(x, frame[layer].to_numpy(), linewidth=0.2, alpha=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.55)
        axis.set_ylabel(layer)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[-1].set_xlabel("Observation index")
    fig.savefig(output_path, dpi=GRID_FIGURE_DPI)
    plt.close(fig)
    return output_path
