"""Example rolling decomposition plots for V2.1 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.config.names import (
    COMPONENT,
    DETAIL_ENERGY_SHARE,
    RMS_VOLATILITY,
)
from multi_scale_volatility.config.paths import (
    FINAL_RETURNS_CSV,
    ROLLING_EXAMPLE_WINDOWS_CSV,
    ROLLING_LAYER_VOLATILITY_CSV,
    ROLLING_PLOTS_DIR,
    ROLLING_RESULTS_DIR,
    ROLLING_SCALE_GROUP_SUMMARY_CSV,
    ROLLING_WINDOW_METADATA_CSV,
    ROLLING_WINDOW_SUMMARY_CSV,
)
from multi_scale_volatility.io import write_csv
from multi_scale_volatility.plotting.save import save_figure
from multi_scale_volatility.plotting.style import (
    FIGURE_DPI,
    FINAL_COLOR,
    FINAL_DARK_COLOR,
    GAUSSIAN_COLOR,
    SHUFFLE_COLOR,
)
from multi_scale_volatility.rolling import (
    ROLLING_K,
    ROLLING_RANDOM_SEED,
    ROLLING_STEP_SIZE,
    decompose_rolling_window_from_input,
)


@dataclass(frozen=True)
class RollingPlotPaths:
    results_dir: Path = ROLLING_RESULTS_DIR
    output_dir: Path = ROLLING_PLOTS_DIR

    @property
    def summary_csv(self) -> Path:
        return self.results_dir / ROLLING_WINDOW_SUMMARY_CSV.name

    @property
    def layer_volatility_csv(self) -> Path:
        return self.results_dir / ROLLING_LAYER_VOLATILITY_CSV.name

    @property
    def scale_group_summary_csv(self) -> Path:
        return self.results_dir / ROLLING_SCALE_GROUP_SUMMARY_CSV.name


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


def create_rolling_plots(paths: RollingPlotPaths | None = None) -> list[Path]:
    paths = paths or RollingPlotPaths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    rms_dir = paths.output_dir / "rms"
    energy_share_dir = paths.output_dir / "energy_share"
    rms_dir.mkdir(parents=True, exist_ok=True)
    energy_share_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(paths.summary_csv)
    layer = pd.read_csv(paths.layer_volatility_csv)
    groups = pd.read_csv(paths.scale_group_summary_csv)

    summary = add_timestamp_column(summary)
    layer = add_timestamp_column(layer)
    groups = add_timestamp_column(groups)

    outputs: list[Path] = []
    window_lengths = sorted(summary["window_length"].unique())
    for window_length in window_lengths:
        outputs.append(
            plot_total_volatility(
                summary,
                paths.output_dir / f"rolling_total_volatility_{window_length}.png",
                window_length=window_length,
            )
        )
        outputs.append(
            plot_layer_heatmap(
                layer,
                rms_dir / f"rms_volatility_heatmap_{window_length}.png",
                window_length=window_length,
                metric=RMS_VOLATILITY,
                title=f"Rolling RMS Volatility Heatmap (W={window_length})",
                colorbar_label="RMS volatility",
                cmap="viridis",
            )
        )
        outputs.append(
            plot_layer_normalized_heatmap(
                layer,
                rms_dir / f"rms_volatility_percentile_heatmap_{window_length}.png",
                window_length=window_length,
                source_metric=RMS_VOLATILITY,
                normalized_metric="percentile",
                title=f"Rolling RMS Volatility Percentile Heatmap (W={window_length})",
                colorbar_label="Within-scale percentile",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
        )
        outputs.append(
            plot_layer_normalized_heatmap(
                layer,
                rms_dir / f"rms_volatility_zscore_heatmap_{window_length}.png",
                window_length=window_length,
                source_metric=RMS_VOLATILITY,
                normalized_metric="zscore",
                title=f"Rolling RMS Volatility Z-Score Heatmap (W={window_length})",
                colorbar_label="Within-scale z-score",
                cmap="coolwarm",
            )
        )
        outputs.append(
            plot_layer_correlation_heatmap(
                layer,
                rms_dir / f"rms_volatility_correlation_{window_length}.png",
                window_length=window_length,
                source_metric=RMS_VOLATILITY,
                normalized_metric=None,
                title=f"Correlation of RMS Volatility by Component (W={window_length})",
            )
        )
        outputs.append(
            plot_layer_correlation_heatmap(
                layer,
                rms_dir / f"rms_volatility_percentile_correlation_{window_length}.png",
                window_length=window_length,
                source_metric=RMS_VOLATILITY,
                normalized_metric="percentile",
                title=(
                    "Correlation of RMS Volatility Percentiles "
                    f"(W={window_length})"
                ),
            )
        )
        outputs.append(
            plot_layer_correlation_heatmap(
                layer,
                rms_dir / f"rms_volatility_zscore_correlation_{window_length}.png",
                window_length=window_length,
                source_metric=RMS_VOLATILITY,
                normalized_metric="zscore",
                title=(
                    "Correlation of RMS Volatility Z-Scores "
                    f"(W={window_length})"
                ),
            )
        )
        outputs.append(
            plot_layer_heatmap(
                layer,
                energy_share_dir / f"detail_energy_share_heatmap_{window_length}.png",
                window_length=window_length,
                metric=DETAIL_ENERGY_SHARE,
                title=f"Rolling Detail Energy Share Heatmap (W={window_length})",
                colorbar_label="Detail energy share",
                cmap="magma",
            )
        )
        outputs.append(
            plot_layer_normalized_heatmap(
                layer,
                energy_share_dir
                / f"detail_energy_share_percentile_heatmap_{window_length}.png",
                window_length=window_length,
                source_metric=DETAIL_ENERGY_SHARE,
                normalized_metric="percentile",
                title=(
                    "Rolling Detail Energy Share Percentile Heatmap "
                    f"(W={window_length})"
                ),
                colorbar_label="Within-scale share percentile",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
        )
        outputs.append(
            plot_detail_share_percentile_correlation(
                layer,
                energy_share_dir
                / f"detail_energy_share_percentile_correlation_{window_length}.png",
                window_length=window_length,
            )
        )
        outputs.append(
            plot_selected_scale_rms(
                layer,
                rms_dir / f"selected_scale_rms_{window_length}.png",
                window_length=window_length,
            )
        )
        outputs.append(
            plot_scale_group_shares(
                groups,
                energy_share_dir / f"fine_mid_coarse_share_{window_length}.png",
                window_length=window_length,
            )
        )
        outputs.append(
            plot_scale_group_share_correlation(
                groups,
                energy_share_dir / f"fine_mid_coarse_share_correlation_{window_length}.png",
                window_length=window_length,
            )
        )

    outputs.append(
        plot_total_volatility_comparison(
            summary,
            paths.output_dir / "window_length_total_volatility_comparison.png",
        )
    )
    for group_name in ["fine", "mid", "coarse"]:
        outputs.append(
        plot_scale_group_comparison(
            groups,
            energy_share_dir / f"window_length_{group_name}_share_comparison.png",
            group_name=group_name,
        )
    )
    return outputs


def add_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["window_end_timestamp"] = pd.to_datetime(
        output["window_end_timestamp_utc"],
        utc=True,
    ).dt.tz_convert(None)
    return output


def plot_total_volatility(
    summary: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    rows = summary[summary["window_length"] == window_length].sort_values("window_id")
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.plot(
        rows["window_end_timestamp"],
        rows["original_rms_volatility"],
        color=FINAL_COLOR,
        linewidth=1.1,
    )
    ax.set_title(f"Rolling Total RMS Volatility (W={window_length})")
    ax.set_xlabel("Window end timestamp")
    ax.set_ylabel("Original RMS volatility")
    format_time_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_total_volatility_comparison(
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    colors = {2048: FINAL_COLOR, 8192: GAUSSIAN_COLOR}
    for window_length, rows in summary.groupby("window_length", sort=True):
        rows = rows.sort_values("window_id")
        ax.plot(
            rows["window_end_timestamp"],
            rows["original_rms_volatility"],
            color=colors.get(int(window_length), FINAL_DARK_COLOR),
            linewidth=1.0,
            label=f"W={int(window_length)}",
        )
    ax.set_title("Rolling Total RMS Volatility by Window Length")
    ax.set_xlabel("Window end timestamp")
    ax.set_ylabel("Original RMS volatility")
    format_time_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_layer_heatmap(
    layer: pd.DataFrame,
    output_path: Path,
    window_length: int,
    metric: str,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> Path:
    components = [f"D_{scale:02d}" for scale in range(1, 10)]
    rows = layer[
        (layer["window_length"] == window_length)
        & (layer[COMPONENT].isin(components))
    ].copy()
    matrix = (
        rows.pivot(index=COMPONENT, columns="window_end_timestamp", values=metric)
        .reindex(components)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    date_values = mdates.date2num(matrix.columns.to_pydatetime())
    extent = [date_values[0], date_values[-1], len(components) - 0.5, -0.5]
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Window end timestamp")
    ax.set_ylabel("Component")
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(components)
    ax.xaxis_date()
    format_time_axis(ax)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_layer_normalized_heatmap(
    layer: pd.DataFrame,
    output_path: Path,
    window_length: int,
    source_metric: str,
    normalized_metric: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    components = [f"D_{scale:02d}" for scale in range(1, 10)]
    rows = layer[
        (layer["window_length"] == window_length)
        & (layer[COMPONENT].isin(components))
    ].copy()
    rows["normalized_value"] = normalize_within_component(
        rows,
        source_metric=source_metric,
        normalized_metric=normalized_metric,
    )
    if normalized_metric == "zscore" and (vmin is None or vmax is None):
        limit = float(np.nanpercentile(np.abs(rows["normalized_value"]), 99))
        limit = max(1.0, limit)
        vmin = -limit
        vmax = limit
    return plot_layer_matrix_heatmap(
        rows,
        output_path,
        components=components,
        metric="normalized_value",
        title=title,
        colorbar_label=colorbar_label,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def normalize_within_component(
    rows: pd.DataFrame,
    source_metric: str,
    normalized_metric: str,
) -> pd.Series:
    grouped = rows.groupby(COMPONENT, sort=False)[source_metric]
    if normalized_metric == "percentile":
        ranks = grouped.rank(method="average") - 1.0
        counts = grouped.transform("count") - 1.0
        return ranks / counts.replace(0.0, np.nan)
    if normalized_metric == "zscore":
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0.0, np.nan)
        return (rows[source_metric] - means) / stds
    raise ValueError(f"Unsupported normalized metric: {normalized_metric}")


def plot_layer_matrix_heatmap(
    rows: pd.DataFrame,
    output_path: Path,
    components: list[str],
    metric: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    matrix = (
        rows.pivot(index=COMPONENT, columns="window_end_timestamp", values=metric)
        .reindex(components)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    date_values = mdates.date2num(matrix.columns.to_pydatetime())
    extent = [date_values[0], date_values[-1], len(components) - 0.5, -0.5]
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Window end timestamp")
    ax.set_ylabel("Component")
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(components)
    ax.xaxis_date()
    format_time_axis(ax)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_detail_share_percentile_correlation(
    layer: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    return plot_layer_correlation_heatmap(
        layer,
        output_path,
        window_length=window_length,
        source_metric=DETAIL_ENERGY_SHARE,
        normalized_metric="percentile",
        title=(
            "Correlation of Detail Energy Share Percentiles "
            f"(W={window_length})"
        ),
    )


def plot_layer_correlation_heatmap(
    layer: pd.DataFrame,
    output_path: Path,
    window_length: int,
    source_metric: str,
    normalized_metric: str | None,
    title: str,
) -> Path:
    components = [f"D_{scale:02d}" for scale in range(1, 10)]
    rows = layer[
        (layer["window_length"] == window_length)
        & (layer[COMPONENT].isin(components))
    ].copy()
    value_column = source_metric
    if normalized_metric is not None:
        value_column = "normalized_value"
        rows[value_column] = normalize_within_component(
            rows,
            source_metric=source_metric,
            normalized_metric=normalized_metric,
        )
    wide = (
        rows.pivot(
            index="window_id",
            columns=COMPONENT,
            values=value_column,
        )
        .reindex(columns=components)
        .sort_index()
    )
    correlation = wide.corr().reindex(index=components, columns=components)
    return plot_correlation_matrix(
        correlation,
        output_path,
        labels=components,
        title=title,
    )


def plot_correlation_matrix(
    correlation: pd.DataFrame,
    output_path: Path,
    labels: list[str],
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    image = ax.imshow(
        correlation.to_numpy(dtype=float),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        aspect="equal",
    )
    values = correlation.to_numpy(dtype=float)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
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
    colorbar.set_label("Pearson correlation")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_selected_scale_rms(
    layer: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    selected_components = ["D_01", "D_03", "D_06", "D_09"]
    colors = {
        "D_01": FINAL_COLOR,
        "D_03": SHUFFLE_COLOR,
        "D_06": GAUSSIAN_COLOR,
        "D_09": FINAL_DARK_COLOR,
    }
    rows = layer[
        (layer["window_length"] == window_length)
        & (layer[COMPONENT].isin(selected_components))
    ].copy()
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for component in selected_components:
        component_rows = rows[rows[COMPONENT] == component].sort_values("window_id")
        ax.plot(
            component_rows["window_end_timestamp"],
            component_rows[RMS_VOLATILITY],
            linewidth=1.0,
            color=colors[component],
            label=component,
        )
    ax.set_title(f"Selected Rolling Component RMS Volatility (W={window_length})")
    ax.set_xlabel("Window end timestamp")
    ax.set_ylabel("RMS volatility")
    format_time_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_scale_group_shares(
    groups: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    rows = groups[groups["window_length"] == window_length].copy()
    return plot_group_lines(
        rows,
        output_path,
        title=f"Fine / Mid / Coarse Detail Energy Share (W={window_length})",
    )


def plot_scale_group_share_correlation(
    groups: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    group_order = ["fine", "mid", "coarse"]
    rows = groups[groups["window_length"] == window_length].copy()
    wide = (
        rows.pivot(
            index="window_id",
            columns="scale_group",
            values="group_detail_energy_share",
        )
        .reindex(columns=group_order)
        .sort_index()
    )
    correlation = wide.corr().reindex(index=group_order, columns=group_order)
    return plot_correlation_matrix(
        correlation,
        output_path,
        labels=group_order,
        title=f"Correlation of Fine / Mid / Coarse Energy Shares (W={window_length})",
    )


def plot_scale_group_comparison(
    groups: pd.DataFrame,
    output_path: Path,
    group_name: str,
) -> Path:
    rows = groups[groups["scale_group"] == group_name].copy()
    fig, ax = plt.subplots(figsize=(13, 5.4))
    colors = {2048: FINAL_COLOR, 8192: GAUSSIAN_COLOR}
    for window_length, group_rows in rows.groupby("window_length", sort=True):
        group_rows = group_rows.sort_values("window_id")
        ax.plot(
            group_rows["window_end_timestamp"],
            group_rows["group_detail_energy_share"],
            linewidth=1.0,
            color=colors.get(int(window_length), FINAL_DARK_COLOR),
            label=f"W={int(window_length)}",
        )
    ax.set_title(f"{group_name.title()} Detail Energy Share by Window Length")
    ax.set_xlabel("Window end timestamp")
    ax.set_ylabel("Group detail energy share")
    ax.set_ylim(0.0, 1.0)
    format_time_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_group_lines(rows: pd.DataFrame, output_path: Path, title: str) -> Path:
    colors = {"fine": FINAL_COLOR, "mid": GAUSSIAN_COLOR, "coarse": SHUFFLE_COLOR}
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for group_name in ["fine", "mid", "coarse"]:
        group_rows = rows[rows["scale_group"] == group_name].sort_values("window_id")
        ax.plot(
            group_rows["window_end_timestamp"],
            group_rows["group_detail_energy_share"],
            linewidth=1.0,
            color=colors[group_name],
            label=group_name,
        )
    ax.set_title(title)
    ax.set_xlabel("Window end timestamp")
    ax.set_ylabel("Group detail energy share")
    ax.set_ylim(0.0, 1.0)
    format_time_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))


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
