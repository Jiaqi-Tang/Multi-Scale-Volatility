"""Example rolling decomposition plots for V2.1 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.config.paths import (
    FINAL_RETURNS_CSV,
    ROLLING_EXAMPLE_WINDOWS_CSV,
    ROLLING_PLOTS_DIR,
    ROLLING_RESULTS_DIR,
    ROLLING_WINDOW_METADATA_CSV,
    ROLLING_WINDOW_SUMMARY_CSV,
)
from multi_scale_volatility.io import write_csv
from multi_scale_volatility.plotting.save import save_figure
from multi_scale_volatility.plotting.style import FIGURE_DPI, FINAL_COLOR
from multi_scale_volatility.rolling import (
    ROLLING_K,
    ROLLING_RANDOM_SEED,
    ROLLING_STEP_SIZE,
    decompose_rolling_window_from_input,
)


@dataclass(frozen=True)
class RollingExamplePlotPaths:
    final_returns_csv: Path = FINAL_RETURNS_CSV
    results_dir: Path = ROLLING_RESULTS_DIR
    output_dir: Path = ROLLING_PLOTS_DIR / "examples"

    @property
    def metadata_csv(self) -> Path:
        return self.results_dir / ROLLING_WINDOW_METADATA_CSV.name

    @property
    def summary_csv(self) -> Path:
        return self.results_dir / ROLLING_WINDOW_SUMMARY_CSV.name

    @property
    def selected_windows_csv(self) -> Path:
        return self.results_dir / ROLLING_EXAMPLE_WINDOWS_CSV.name


def create_rolling_example_decomposition_plots(
    paths: RollingExamplePlotPaths | None = None,
    k: int = ROLLING_K,
    step_size: int = ROLLING_STEP_SIZE,
    random_seed: int = ROLLING_RANDOM_SEED,
    random_windows_per_length: int = 3,
) -> list[Path]:
    paths = paths or RollingExamplePlotPaths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(paths.metadata_csv)
    summary = pd.read_csv(paths.summary_csv)
    selected = select_example_windows(
        metadata,
        summary,
        random_seed=random_seed,
        random_windows_per_length=random_windows_per_length,
    )
    write_csv(selected, paths.selected_windows_csv, index=False)

    outputs: list[Path] = []
    for row in selected.to_dict("records"):
        window_length = int(row["window_length"])
        window_id = int(row["window_id"])
        reason = str(row["selection_reason"])
        frame = decompose_rolling_window_from_input(
            paths.final_returns_csv,
            window_length=window_length,
            window_id=window_id,
            step_size=step_size,
            k=k,
        )
        safe_reason = reason.replace("_", "-")
        output_path = (
            paths.output_dir
            / f"rolling_decomposition_w{window_length}_id{window_id:04d}_{safe_reason}.png"
        )
        outputs.append(
            plot_rolling_decomposition_example(
                frame,
                output_path,
                title=(
                    f"Rolling Decomposition: W={window_length}, "
                    f"id={window_id}, {reason.replace('_', ' ')}"
                ),
            )
        )
    return outputs


def select_example_windows(
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    random_seed: int,
    random_windows_per_length: int,
) -> pd.DataFrame:
    joined = metadata.merge(
        summary[
            [
                "window_length",
                "window_id",
                "original_rms_volatility",
                "max_abs_reconstruction_error",
            ]
        ],
        on=["window_length", "window_id"],
        how="inner",
    )
    if joined.empty:
        raise ValueError("No rolling windows available for example selection")

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for window_length, group in joined.groupby("window_length", sort=True):
        group = group.sort_values("window_id").reset_index(drop=True)
        add_selection(rows, group.iloc[0], "first_window")
        add_selection(rows, group.iloc[-1], "last_window")
        add_selection(
            rows,
            group.loc[group["original_rms_volatility"].idxmin()],
            "min_total_vol",
        )
        add_selection(
            rows,
            group.loc[group["original_rms_volatility"].idxmax()],
            "max_total_vol",
        )

        median_vol = float(group["original_rms_volatility"].median())
        median_index = (
            group["original_rms_volatility"].sub(median_vol).abs().idxmin()
        )
        add_selection(rows, group.loc[median_index], "median_total_vol")

        sample_size = min(random_windows_per_length, len(group))
        random_indices = rng.choice(group.index.to_numpy(), size=sample_size, replace=False)
        for random_number, index in enumerate(sorted(random_indices), start=1):
            add_selection(rows, group.loc[index], f"random_{random_number}")

    selected = pd.DataFrame(rows)
    selected = selected.drop_duplicates(
        subset=["window_length", "window_id"],
        keep="first",
    )
    return selected.sort_values(["window_length", "window_id"]).reset_index(drop=True)


def add_selection(rows: list[dict[str, Any]], row: pd.Series, reason: str) -> None:
    output = row.to_dict()
    output["selection_reason"] = reason
    rows.append(output)


def plot_rolling_decomposition_example(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> Path:
    layers = [
        ("original", "Original"),
        *[(f"D_{scale:02d}", f"D_{scale:02d}") for scale in range(1, 10)],
        ("A_09", "A_09"),
    ]
    x = frame["index"].to_numpy()
    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(14, 15.5),
        sharex=True,
        constrained_layout=True,
    )
    for axis, (column, label) in zip(axes, layers, strict=True):
        values = frame[column].astype(float).to_numpy()
        axis.plot(
            x,
            values,
            color=FINAL_COLOR,
            linewidth=0.45,
            alpha=0.85,
            rasterized=True,
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.65)
        axis.set_ylabel(label)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[-1].set_xlabel("Window-local observation index")
    fig.suptitle(title)
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path
