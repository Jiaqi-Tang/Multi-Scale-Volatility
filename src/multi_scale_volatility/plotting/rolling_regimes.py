"""Plots for V2.3 rolling volatility-state maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.config.names import (
    COMPONENT,
    COMPONENT_TYPE,
    DETAIL_ENERGY_SHARE,
    LOG_RETURN,
    RMS_VOLATILITY,
)
from multi_scale_volatility.config.paths import (
    FINAL_RETURNS_CSV,
    ROLLING_LAYER_VOLATILITY_CSV,
    ROLLING_REGIME_CELL_COUNTS_CSV,
    ROLLING_REGIME_EPISODE_SUMMARY_CSV,
    ROLLING_REGIME_METRICS_CSV,
    ROLLING_REGIME_PLOTS_DIR,
    ROLLING_REGIME_RESULTS_DIR,
    ROLLING_WINDOW_METADATA_CSV,
)
from multi_scale_volatility.plotting.save import save_figure
from multi_scale_volatility.plotting.style import (
    FIGURE_DPI,
    FINAL_COLOR,
    FINAL_DARK_COLOR,
    GAUSSIAN_COLOR,
    SHUFFLE_COLOR,
)
from multi_scale_volatility.rolling_regimes import (
    FINE_BUCKET_ORDER,
    PROFILE_REGIMES,
    VOL_BUCKET_ORDER,
)


@dataclass(frozen=True)
class RollingRegimePlotPaths:
    final_returns_csv: Path = FINAL_RETURNS_CSV
    results_dir: Path = ROLLING_REGIME_RESULTS_DIR
    rolling_results_dir: Path = ROLLING_REGIME_RESULTS_DIR.parent
    output_dir: Path = ROLLING_REGIME_PLOTS_DIR

    @property
    def regime_metrics_csv(self) -> Path:
        return self.results_dir / ROLLING_REGIME_METRICS_CSV.name

    @property
    def episode_summary_csv(self) -> Path:
        return self.results_dir / ROLLING_REGIME_EPISODE_SUMMARY_CSV.name

    @property
    def cell_counts_csv(self) -> Path:
        return self.results_dir / ROLLING_REGIME_CELL_COUNTS_CSV.name

    @property
    def layer_volatility_csv(self) -> Path:
        return self.rolling_results_dir / ROLLING_LAYER_VOLATILITY_CSV.name

    @property
    def metadata_csv(self) -> Path:
        return self.rolling_results_dir / ROLLING_WINDOW_METADATA_CSV.name


def create_rolling_regime_plots(
    paths: RollingRegimePlotPaths | None = None,
) -> list[Path]:
    paths = paths or RollingRegimePlotPaths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(paths.regime_metrics_csv)
    episodes = pd.read_csv(paths.episode_summary_csv)
    counts = pd.read_csv(paths.cell_counts_csv)
    layer = pd.read_csv(paths.layer_volatility_csv)
    metadata = pd.read_csv(paths.metadata_csv)
    prices = pd.read_csv(
        paths.final_returns_csv,
        usecols=["timestamp_utc", "close", LOG_RETURN],
    )

    metrics["window_end_dt"] = pd.to_datetime(
        metrics["window_end_timestamp_utc"], utc=True
    )
    prices["timestamp_dt"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
    trend_ratio = compute_window_trend_ratios(prices, metadata)
    trend_ratio["window_end_dt"] = pd.to_datetime(
        trend_ratio["window_end_timestamp_utc"], utc=True
    )
    metrics["time_num"] = mdates.date2num(
        metrics["window_end_dt"].dt.tz_convert(None).dt.to_pydatetime()
    )
    if not episodes.empty:
        episodes["start_dt"] = pd.to_datetime(episodes["start_timestamp_utc"], utc=True)
        episodes["end_dt"] = pd.to_datetime(episodes["end_timestamp_utc"], utc=True)

    outputs: list[Path] = []
    for window_length in sorted(metrics["window_length"].unique()):
        rows = metrics[metrics["window_length"] == window_length].copy()
        outputs.append(
            plot_regime_scatter_percentile(
                rows,
                paths.output_dir / f"regime_scatter_percentile_{window_length}.png",
                window_length=int(window_length),
            )
        )
        outputs.append(
            plot_regime_scatter_raw_rms(
                rows,
                paths.output_dir / f"regime_scatter_raw_rms_{window_length}.png",
                window_length=int(window_length),
            )
        )
        outputs.append(
            plot_cell_counts(
                counts[counts["window_length"] == window_length],
                paths.output_dir / f"regime_cell_counts_{window_length}.png",
                window_length=int(window_length),
            )
        )
        outputs.append(
            plot_regime_metric_summary(
                rows,
                paths.output_dir / f"regime_total_rms_summary_{window_length}.png",
                window_length=int(window_length),
                metric="original_rms_volatility",
                title=f"Total RMS Summary by Regime (W={window_length})",
                colorbar_label="Mean total RMS",
            )
        )
        outputs.append(
            plot_regime_metric_summary(
                rows,
                paths.output_dir / f"regime_fine_share_summary_{window_length}.png",
                window_length=int(window_length),
                metric="fine_detail_energy_share",
                title=f"Fine Share Summary by Regime (W={window_length})",
                colorbar_label="Mean fine detail energy share",
            )
        )
        outputs.append(
            plot_episode_duration_summary(
                episodes[episodes["window_length"] == window_length],
                paths.output_dir / f"regime_episode_duration_summary_{window_length}.png",
                window_length=int(window_length),
            )
        )
        outputs.append(
            plot_total_rms_with_regime_shading(
                rows,
                episodes[episodes["window_length"] == window_length],
                paths.output_dir
                / f"total_rms_with_highvol_regimes_{window_length}.png",
                window_length=int(window_length),
            )
        )
        outputs.append(
            plot_price_with_regime_shading(
                prices,
                episodes[episodes["window_length"] == window_length],
                paths.output_dir
                / f"raw_price_with_highvol_regimes_{window_length}.png",
                window_length=int(window_length),
                vol_bucket="highVol",
            )
        )
        outputs.append(
            plot_trend_ratio_with_regime_shading(
                trend_ratio[trend_ratio["window_length"] == window_length],
                episodes[episodes["window_length"] == window_length],
                paths.output_dir
                / f"trend_ratio_with_highvol_regimes_{window_length}.png",
                window_length=int(window_length),
            )
        )
        outputs.append(
            plot_price_with_regime_shading(
                prices,
                episodes[episodes["window_length"] == window_length],
                paths.output_dir
                / f"raw_price_with_lowvol_regimes_{window_length}.png",
                window_length=int(window_length),
                vol_bucket="lowVol",
            )
        )
        outputs.append(
            plot_average_profiles(
                layer,
                metrics,
                paths.output_dir / f"regime_average_rms_profiles_{window_length}.png",
                window_length=int(window_length),
                metric=RMS_VOLATILITY,
                ylabel="RMS volatility",
                title=f"Average RMS Profiles by Regime (W={window_length})",
            )
        )
        outputs.append(
            plot_average_profiles(
                layer,
                metrics,
                paths.output_dir
                / f"regime_average_energy_share_profiles_{window_length}.png",
                window_length=int(window_length),
                metric=DETAIL_ENERGY_SHARE,
                ylabel="Detail energy share",
                title=f"Average Detail Energy Share Profiles by Regime (W={window_length})",
            )
        )
    return outputs


def plot_regime_scatter_percentile(
    rows: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    scatter = ax.scatter(
        rows["total_rms_percentile"],
        rows["fine_share_percentile"],
        c=rows["time_num"],
        cmap="viridis",
        s=14,
        alpha=0.62,
        linewidths=0,
    )
    draw_regime_grid(ax)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Total RMS percentile")
    ax.set_ylabel("Fine-share percentile")
    ax.set_title(f"Rolling Volatility-State Map (W={window_length})")
    add_year_colorbar(fig, ax, scatter, rows["window_end_dt"])
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def plot_regime_scatter_raw_rms(
    rows: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    lower = rows["original_rms_volatility"].quantile(0.20, interpolation="linear")
    upper = rows["original_rms_volatility"].quantile(0.80, interpolation="linear")
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    scatter = ax.scatter(
        rows["original_rms_volatility"],
        rows["fine_share_percentile"],
        c=rows["time_num"],
        cmap="viridis",
        s=14,
        alpha=0.62,
        linewidths=0,
    )
    for x_value in [lower, upper]:
        ax.axvline(x_value, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    for y_value in [0.20, 0.80]:
        ax.axhline(y_value, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Raw total RMS volatility")
    ax.set_ylabel("Fine-share percentile")
    ax.set_title(f"Raw RMS Volatility-State Map (W={window_length})")
    add_year_colorbar(fig, ax, scatter, rows["window_end_dt"])
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def plot_cell_counts(rows: pd.DataFrame, output_path: Path, window_length: int) -> Path:
    matrix = rows.pivot(
        index="fine_bucket",
        columns="vol_bucket",
        values="window_count",
    ).reindex(index=FINE_BUCKET_ORDER, columns=VOL_BUCKET_ORDER)
    share_matrix = rows.pivot(
        index="fine_bucket",
        columns="vol_bucket",
        values="window_share",
    ).reindex(index=FINE_BUCKET_ORDER, columns=VOL_BUCKET_ORDER)

    fig, ax = plt.subplots(figsize=(7.7, 6.6))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(VOL_BUCKET_ORDER)), labels=VOL_BUCKET_ORDER)
    ax.set_yticks(np.arange(len(FINE_BUCKET_ORDER)), labels=FINE_BUCKET_ORDER)
    ax.set_xlabel("Volatility bucket")
    ax.set_ylabel("Fine-share bucket")
    ax.set_title(f"Regime Cell Counts (W={window_length})")
    for row_index, fine_bucket in enumerate(FINE_BUCKET_ORDER):
        for col_index, vol_bucket in enumerate(VOL_BUCKET_ORDER):
            count = int(matrix.loc[fine_bucket, vol_bucket])
            share = float(share_matrix.loc[fine_bucket, vol_bucket])
            ax.text(
                col_index,
                row_index,
                f"{count}\n{share:.1%}",
                ha="center",
                va="center",
                color=annotation_color(image, count),
                fontsize=10,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Window count")
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def plot_total_rms_with_regime_shading(
    rows: pd.DataFrame,
    episodes: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    x_values = rows["window_end_dt"].dt.tz_convert(None)
    ax.plot(
        x_values,
        rows["original_rms_volatility"],
        color=FINAL_DARK_COLOR,
        linewidth=1.0,
        label="Total RMS",
    )
    selected = episodes[
        episodes["regime_label"].isin(["highVol_highFine", "highVol_lowFine"])
    ]
    colors = {
        "highVol_highFine": GAUSSIAN_COLOR,
        "highVol_lowFine": SHUFFLE_COLOR,
    }
    labels_seen: set[str] = set()
    for _, episode in selected.iterrows():
        label = str(episode["regime_label"])
        start = episode["start_dt"].tz_convert(None)
        end = episode["end_dt"].tz_convert(None)
        ax.axvspan(
            start,
            end,
            color=colors[label],
            alpha=0.22,
            label=label if label not in labels_seen else None,
        )
        labels_seen.add(label)
    ax.set_xlabel("Window end year")
    ax.set_ylabel("Raw total RMS volatility")
    ax.set_title(f"Total RMS with High-Volatility Regime Episodes (W={window_length})")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def compute_window_trend_ratios(
    returns: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    values = returns[LOG_RETURN].astype(float).to_numpy()
    rows: list[dict[str, float | int | str]] = []
    for row in metadata.itertuples(index=False):
        start = int(row.window_start_index)
        end = int(row.window_end_index)
        window = values[start : end + 1]
        denominator = float(np.sum(np.abs(window)))
        if denominator == 0.0:
            trend_ratio = np.nan
        else:
            trend_ratio = float(abs(np.sum(window)) / denominator)
        rows.append(
            {
                "window_length": int(row.window_length),
                "window_id": int(row.window_id),
                "window_end_timestamp_utc": row.window_end_timestamp_utc,
                "trend_ratio": trend_ratio,
            }
        )
    return pd.DataFrame(rows)


def plot_trend_ratio_with_regime_shading(
    rows: pd.DataFrame,
    episodes: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.plot(
        rows["window_end_dt"].dt.tz_convert(None),
        rows["trend_ratio"],
        color=FINAL_DARK_COLOR,
        linewidth=1.0,
        label="Trend ratio",
    )
    selected = episodes[
        episodes["regime_label"].isin(
            ["highVol_lowFine", "highVol_midFine", "highVol_highFine"]
        )
    ]
    colors = {
        "highVol_lowFine": SHUFFLE_COLOR,
        "highVol_midFine": "#8a7f3a",
        "highVol_highFine": GAUSSIAN_COLOR,
    }
    labels_seen: set[str] = set()
    for _, episode in selected.iterrows():
        label = str(episode["regime_label"])
        start = episode["start_dt"].tz_convert(None)
        end = episode["end_dt"].tz_convert(None)
        ax.axvspan(
            start,
            end,
            color=colors[label],
            alpha=0.20,
            label=label if label not in labels_seen else None,
        )
        labels_seen.add(label)
    ax.set_xlabel("Window end year")
    ax.set_ylabel("Trend ratio")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Trend Ratio with High-Volatility Regime Episodes (W={window_length})")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, ncols=2)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def plot_regime_metric_summary(
    rows: pd.DataFrame,
    output_path: Path,
    window_length: int,
    metric: str,
    title: str,
    colorbar_label: str,
) -> Path:
    summary = rows.groupby(["fine_bucket", "vol_bucket"])[metric].agg(
        mean="mean",
        median="median",
        std="std",
    )
    mean_matrix = summary["mean"].unstack("vol_bucket").reindex(
        index=FINE_BUCKET_ORDER,
        columns=VOL_BUCKET_ORDER,
    )
    median_matrix = summary["median"].unstack("vol_bucket").reindex(
        index=FINE_BUCKET_ORDER,
        columns=VOL_BUCKET_ORDER,
    )
    std_matrix = summary["std"].unstack("vol_bucket").reindex(
        index=FINE_BUCKET_ORDER,
        columns=VOL_BUCKET_ORDER,
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.9))
    image = ax.imshow(mean_matrix.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(VOL_BUCKET_ORDER)), labels=VOL_BUCKET_ORDER)
    ax.set_yticks(np.arange(len(FINE_BUCKET_ORDER)), labels=FINE_BUCKET_ORDER)
    ax.set_xlabel("Volatility bucket")
    ax.set_ylabel("Fine-share bucket")
    ax.set_title(title)
    for row_index, fine_bucket in enumerate(FINE_BUCKET_ORDER):
        for col_index, vol_bucket in enumerate(VOL_BUCKET_ORDER):
            mean_value = float(mean_matrix.loc[fine_bucket, vol_bucket])
            median_value = float(median_matrix.loc[fine_bucket, vol_bucket])
            std_value = float(std_matrix.loc[fine_bucket, vol_bucket])
            ax.text(
                col_index,
                row_index,
                f"mean {mean_value:.2e}\nmed {median_value:.2e}\nstd {std_value:.2e}",
                ha="center",
                va="center",
                color=annotation_color(image, mean_value),
                fontsize=8.6,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def plot_episode_duration_summary(
    episodes: pd.DataFrame,
    output_path: Path,
    window_length: int,
) -> Path:
    summary = episodes.groupby("regime_label")["duration_windows"].agg(
        mean="mean",
        count="count",
    )
    mean_matrix = pd.DataFrame(
        np.nan,
        index=FINE_BUCKET_ORDER,
        columns=VOL_BUCKET_ORDER,
        dtype=float,
    )
    count_matrix = pd.DataFrame(
        0,
        index=FINE_BUCKET_ORDER,
        columns=VOL_BUCKET_ORDER,
        dtype=int,
    )
    for fine_bucket in FINE_BUCKET_ORDER:
        for vol_bucket in VOL_BUCKET_ORDER:
            regime_label = f"{vol_bucket}_{fine_bucket}"
            if regime_label in summary.index:
                mean_matrix.loc[fine_bucket, vol_bucket] = float(
                    summary.loc[regime_label, "mean"]
                )
                count_matrix.loc[fine_bucket, vol_bucket] = int(
                    summary.loc[regime_label, "count"]
                )

    fig, ax = plt.subplots(figsize=(8.2, 6.9))
    image = ax.imshow(
        np.ma.masked_invalid(mean_matrix.to_numpy(dtype=float)),
        cmap="Purples",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(VOL_BUCKET_ORDER)), labels=VOL_BUCKET_ORDER)
    ax.set_yticks(np.arange(len(FINE_BUCKET_ORDER)), labels=FINE_BUCKET_ORDER)
    ax.set_xlabel("Volatility bucket")
    ax.set_ylabel("Fine-share bucket")
    ax.set_title(f"Average Episode Duration by Regime (W={window_length})")
    for row_index, fine_bucket in enumerate(FINE_BUCKET_ORDER):
        for col_index, vol_bucket in enumerate(VOL_BUCKET_ORDER):
            mean_value = mean_matrix.loc[fine_bucket, vol_bucket]
            count = int(count_matrix.loc[fine_bucket, vol_bucket])
            if pd.isna(mean_value):
                label = "no episodes"
                color = "black"
            else:
                label = f"mean {mean_value:.1f} steps\nn {count}"
                color = annotation_color(image, float(mean_value))
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=9.2,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean duration, rolling steps")
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def annotation_color(image: plt.AxesImage, value: float) -> str:
    rgba = image.cmap(image.norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.48 else "black"


def plot_price_with_regime_shading(
    prices: pd.DataFrame,
    episodes: pd.DataFrame,
    output_path: Path,
    window_length: int,
    vol_bucket: str,
) -> Path:
    selected_regimes = [
        f"{vol_bucket}_lowFine",
        f"{vol_bucket}_midFine",
        f"{vol_bucket}_highFine",
    ]
    title_prefix = "High-Volatility" if vol_bucket == "highVol" else "Low-Volatility"
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.plot(
        prices["timestamp_dt"].dt.tz_convert(None),
        prices["close"],
        color=FINAL_DARK_COLOR,
        linewidth=0.65,
        label="EUR/USD close",
    )
    selected = episodes[episodes["regime_label"].isin(selected_regimes)]
    colors = {
        f"{vol_bucket}_lowFine": SHUFFLE_COLOR,
        f"{vol_bucket}_midFine": "#8a7f3a",
        f"{vol_bucket}_highFine": GAUSSIAN_COLOR,
    }
    labels_seen: set[str] = set()
    for _, episode in selected.iterrows():
        label = str(episode["regime_label"])
        start = episode["start_dt"].tz_convert(None)
        end = episode["end_dt"].tz_convert(None)
        ax.axvspan(
            start,
            end,
            color=colors[label],
            alpha=0.20,
            label=label if label not in labels_seen else None,
        )
        labels_seen.add(label)
    ax.set_xlabel("Year")
    ax.set_ylabel("EUR/USD close")
    ax.set_title(f"Raw Price with {title_prefix} Regime Episodes (W={window_length})")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, ncols=2)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def plot_average_profiles(
    layer: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
    window_length: int,
    metric: str,
    ylabel: str,
    title: str,
) -> Path:
    regime_lookup = metrics[
        ["window_length", "window_id", "regime_label"]
    ].drop_duplicates()
    rows = layer[
        (layer["window_length"] == window_length)
        & (layer[COMPONENT_TYPE] == "detail")
    ].merge(
        regime_lookup,
        on=["window_length", "window_id"],
        how="inner",
        validate="many_to_one",
    )
    rows = rows[rows["regime_label"].isin(PROFILE_REGIMES)].copy()
    profile = (
        rows.groupby(["regime_label", COMPONENT], as_index=False)[metric]
        .mean()
        .sort_values([COMPONENT, "regime_label"])
    )
    components = sorted(profile[COMPONENT].unique())
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = {
        "highVol_highFine": FINAL_DARK_COLOR,
        "highVol_lowFine": GAUSSIAN_COLOR,
        "lowVol_highFine": FINAL_COLOR,
        "lowVol_lowFine": SHUFFLE_COLOR,
    }
    for regime in PROFILE_REGIMES:
        regime_rows = profile[profile["regime_label"] == regime].set_index(COMPONENT)
        if regime_rows.empty:
            continue
        ax.plot(
            components,
            regime_rows.reindex(components)[metric],
            marker="o",
            linewidth=1.6,
            label=regime,
            color=colors[regime],
        )
    ax.set_xlabel("Detail component")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    saved = save_figure(fig, output_path, FIGURE_DPI)
    plt.close(fig)
    return saved


def draw_regime_grid(ax: plt.Axes) -> None:
    for value in [0.20, 0.80]:
        ax.axvline(value, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(value, color="black", linewidth=1.0, linestyle="--", alpha=0.7)


def add_year_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    scatter: plt.Collection,
    timestamps: pd.Series,
) -> None:
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    years = sorted(int(year) for year in timestamps.dt.year.unique())
    year_ticks = [
        mdates.date2num(pd.Timestamp(year=year, month=1, day=1).to_pydatetime())
        for year in years
    ]
    colorbar.set_ticks(year_ticks)
    colorbar.set_ticklabels([str(year) for year in years])
    colorbar.set_label("Window end year")
