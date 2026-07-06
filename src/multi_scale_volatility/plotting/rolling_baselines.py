"""Plots for rolling Monte Carlo baseline correlation envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.config.paths import (
    ROLLING_BASELINE_CORRELATION_EMPIRICAL_COMPARISON_CSV,
    ROLLING_BASELINE_PLOTS_DIR,
    ROLLING_BASELINE_RESULTS_DIR,
)
from multi_scale_volatility.plotting.save import save_figure
from multi_scale_volatility.plotting.style import FIGURE_DPI

CORRELATION_KIND_DIRS = {
    "rms_volatility_correlation": "rms",
    "detail_energy_share_percentile_correlation": "energy_share",
}

CORRELATION_KIND_LABELS = {
    "rms_volatility_correlation": "RMS Volatility Correlation",
    "detail_energy_share_percentile_correlation": (
        "Detail Energy Share Percentile Correlation"
    ),
}


@dataclass(frozen=True)
class RollingBaselinePlotPaths:
    results_dir: Path = ROLLING_BASELINE_RESULTS_DIR
    output_dir: Path = ROLLING_BASELINE_PLOTS_DIR

    @property
    def comparison_csv(self) -> Path:
        return self.results_dir / ROLLING_BASELINE_CORRELATION_EMPIRICAL_COMPARISON_CSV.name


def create_rolling_baseline_plots(
    paths: RollingBaselinePlotPaths | None = None,
) -> list[Path]:
    paths = paths or RollingBaselinePlotPaths()
    comparison = pd.read_csv(paths.comparison_csv)
    comparison = comparison[comparison["metric"] == "correlation"].copy()
    outputs: list[Path] = []
    for correlation_kind in sorted(comparison["correlation_kind"].unique()):
        output_dir = paths.output_dir / CORRELATION_KIND_DIRS.get(
            correlation_kind,
            correlation_kind,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for window_length in sorted(comparison["window_length"].unique()):
            rows = comparison[
                (comparison["correlation_kind"] == correlation_kind)
                & (comparison["window_length"] == window_length)
            ].copy()
            labels = sorted(rows["component_i"].unique())
            title_base = f"{CORRELATION_KIND_LABELS[correlation_kind]} (W={window_length})"
            outputs.append(
                plot_comparison_matrix(
                    rows[rows["baseline_type"] == rows["baseline_type"].iloc[0]],
                    output_dir / f"{correlation_kind}_empirical_{window_length}.png",
                    labels=labels,
                    value_column="empirical_value",
                    title=f"Empirical {title_base}",
                    vmin=-1.0,
                    vmax=1.0,
                    cmap="coolwarm",
                    colorbar_label="Pearson correlation",
                )
            )
            for baseline_type in ["shuffle", "gaussian"]:
                baseline_rows = rows[rows["baseline_type"] == baseline_type].copy()
                outputs.append(
                    plot_comparison_matrix(
                        baseline_rows,
                        output_dir
                        / f"{correlation_kind}_{baseline_type}_median_{window_length}.png",
                        labels=labels,
                        value_column="baseline_median",
                        title=f"{baseline_type.title()} Median {title_base}",
                        vmin=-1.0,
                        vmax=1.0,
                        cmap="coolwarm",
                        colorbar_label="Pearson correlation",
                    )
                )
                diff_limit = max(
                    0.05,
                    float(np.nanmax(np.abs(baseline_rows["difference_from_median"]))),
                )
                outputs.append(
                    plot_comparison_matrix(
                        baseline_rows,
                        output_dir
                        / f"{correlation_kind}_empirical_minus_{baseline_type}_median_{window_length}.png",
                        labels=labels,
                        value_column="difference_from_median",
                        title=f"Empirical - {baseline_type.title()} Median {title_base}",
                        vmin=-diff_limit,
                        vmax=diff_limit,
                        cmap="coolwarm",
                        colorbar_label="Difference from median",
                    )
                )
                outside = baseline_rows.copy()
                outside["outside_envelope"] = outside["outside_envelope"].astype(bool).astype(float)
                outputs.append(
                    plot_comparison_matrix(
                        outside,
                        output_dir
                        / f"{correlation_kind}_outside_{baseline_type}_envelope_{window_length}.png",
                        labels=labels,
                        value_column="outside_envelope",
                        title=f"Outside {baseline_type.title()} 5-95% Envelope {title_base}",
                        vmin=0.0,
                        vmax=1.0,
                        cmap="Reds",
                        colorbar_label="Outside envelope",
                        annotate_as_integer=True,
                    )
                )
    return outputs


def plot_comparison_matrix(
    rows: pd.DataFrame,
    output_path: Path,
    labels: list[str],
    value_column: str,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
    annotate_as_integer: bool = False,
) -> Path:
    matrix = (
        rows.pivot(index="component_i", columns="component_j", values=value_column)
        .reindex(index=labels, columns=labels)
    )
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isnan(value):
                label = ""
            elif annotate_as_integer:
                label = f"{int(value)}"
            else:
                label = f"{value:.2f}"
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path
