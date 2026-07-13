"""Exploratory plots for V3 volatility-spike events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.core.config.paths import (
    EVENT_CATALOG_CSV,
    EVENT_EDA_PLOTS_DIR,
    EVENT_STUDY_RESULTS_DIR,
    EVENT_WINDOWS_PARQUET,
)
from multi_scale_volatility.core.io import write_csv
from multi_scale_volatility.plotting.save import save_figure
from multi_scale_volatility.plotting.style import FIGURE_DPI, FINAL_COLOR, FINAL_DARK_COLOR
from multi_scale_volatility.research.event_study.events import PRIMARY_WINDOW, trailing_rms

EVENT_EDA_RANDOM_SEED = 20260712
OBSERVATIONS_PER_INDEXED_DAY = 288
WEEKDAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class EventEdaPlotPaths:
    results_dir: Path = EVENT_STUDY_RESULTS_DIR
    output_dir: Path = EVENT_EDA_PLOTS_DIR

    @property
    def catalog_csv(self) -> Path:
        return self.results_dir / EVENT_CATALOG_CSV.name

    @property
    def windows_parquet(self) -> Path:
        return self.results_dir / EVENT_WINDOWS_PARQUET.name


def create_event_eda_plots(
    paths: EventEdaPlotPaths | None = None,
    random_seed: int = EVENT_EDA_RANDOM_SEED,
    random_event_count: int = 3,
) -> list[Path]:
    """Create timing, example decomposition, RMS, and trigger-scatter plots."""
    paths = paths or EventEdaPlotPaths()
    if random_event_count < 0:
        raise ValueError("random_event_count must be non-negative")
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = paths.output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(paths.catalog_csv)
    windows = pd.read_parquet(paths.windows_parquet)
    if catalog.empty or windows.empty:
        raise ValueError("Event catalog and event windows must be non-empty")
    catalog = add_event_calendar_fields(catalog)

    weekday_hour = build_weekday_hour_counts(catalog)
    annual = build_annual_event_counts(catalog)
    intervals = build_consecutive_event_intervals(catalog)
    selected = select_example_events(catalog, random_seed, random_event_count)
    scatter = build_pre_event_volatility_table(catalog, windows)

    weekday_csv = paths.output_dir / "event_weekday_hour_counts.csv"
    annual_csv = paths.output_dir / "event_counts_by_year.csv"
    selected_csv = paths.output_dir / "selected_example_events.csv"
    scatter_csv = paths.output_dir / "pre_event_vs_trigger_rms.csv"
    intervals_csv = paths.output_dir / "consecutive_event_intervals.csv"
    write_csv(weekday_hour, weekday_csv, index=False)
    write_csv(annual, annual_csv, index=False)
    write_csv(selected, selected_csv, index=False)
    write_csv(scatter, scatter_csv, index=False)
    write_csv(intervals, intervals_csv, index=False)

    outputs = [
        plot_weekday_hour_counts(
            weekday_hour, paths.output_dir / "event_timing_weekday_hour_counts.png"
        ),
        plot_annual_event_counts(
            annual, paths.output_dir / "event_counts_by_year.png"
        ),
        plot_pre_event_scatter(
            scatter, paths.output_dir / "pre_event_vs_trigger_rms_scatter.png"
        ),
        plot_consecutive_event_intervals(
            intervals, paths.output_dir / "consecutive_event_interval_histogram.png"
        ),
    ]
    for event in selected.itertuples(index=False):
        event_frame = windows[windows["event_id"] == event.event_id].sort_values(
            "relative_observation"
        )
        label = str(event.selection_reason).replace("_", "-")
        stem = f"event_{int(event.event_id):03d}_{label}"
        title_suffix = f"event {int(event.event_id)}, {event.event_timestamp_utc}"
        outputs.append(
            plot_event_decomposition(
                event_frame,
                examples_dir / f"{stem}_decomposition.png",
                f"Event-window decomposition: {title_suffix}",
            )
        )
        outputs.append(
            plot_event_rms(
                event_frame,
                examples_dir / f"{stem}_w80_rms.png",
                f"Trailing 80-minute RMS: {title_suffix}",
            )
        )
    return [weekday_csv, annual_csv, selected_csv, scatter_csv, intervals_csv, *outputs]


def add_event_calendar_fields(catalog: pd.DataFrame) -> pd.DataFrame:
    output = catalog.copy()
    timestamps = pd.to_datetime(output["event_timestamp_utc"], utc=True)
    output["weekday"] = timestamps.dt.day_name()
    output["weekday_number"] = timestamps.dt.weekday
    output["utc_hour"] = timestamps.dt.hour
    output["year"] = timestamps.dt.year
    return output


def build_weekday_hour_counts(catalog: pd.DataFrame) -> pd.DataFrame:
    counts = catalog.groupby(["weekday_number", "weekday", "utc_hour"]).size()
    rows: list[dict[str, Any]] = []
    for weekday_number, weekday in enumerate(WEEKDAY_ORDER):
        for hour in range(24):
            rows.append({
                "weekday_number": weekday_number,
                "weekday": weekday,
                "utc_hour": hour,
                "event_count": int(counts.get((weekday_number, weekday, hour), 0)),
            })
    return pd.DataFrame(rows)


def build_annual_event_counts(catalog: pd.DataFrame) -> pd.DataFrame:
    start, end = int(catalog["year"].min()), int(catalog["year"].max())
    counts = catalog.groupby("year").size()
    return pd.DataFrame({
        "year": np.arange(start, end + 1),
        "event_count": [int(counts.get(year, 0)) for year in range(start, end + 1)],
    })


def build_consecutive_event_intervals(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return one observation-index and calendar interval after each first event."""
    ordered = catalog.sort_values(["anchor_index", "event_id"]).reset_index(drop=True)
    if len(ordered) < 2:
        return pd.DataFrame(columns=[
            "previous_event_id", "event_id", "previous_event_timestamp_utc",
            "event_timestamp_utc", "interval_observations", "interval_indexed_days",
            "calendar_elapsed_hours",
        ])
    timestamps = pd.to_datetime(ordered["event_timestamp_utc"], utc=True)
    intervals = pd.DataFrame({
        "previous_event_id": ordered["event_id"].shift(1).iloc[1:].astype(int).to_numpy(),
        "event_id": ordered["event_id"].iloc[1:].astype(int).to_numpy(),
        "previous_event_timestamp_utc": ordered["event_timestamp_utc"].shift(1).iloc[1:].to_numpy(),
        "event_timestamp_utc": ordered["event_timestamp_utc"].iloc[1:].to_numpy(),
        "interval_observations": ordered["anchor_index"].diff().iloc[1:].astype(int).to_numpy(),
    })
    intervals["interval_indexed_days"] = (
        intervals["interval_observations"] / OBSERVATIONS_PER_INDEXED_DAY
    )
    intervals["calendar_elapsed_hours"] = (
        timestamps.diff().iloc[1:].dt.total_seconds().to_numpy() / 3600
    )
    return intervals


def select_example_events(
    catalog: pd.DataFrame,
    random_seed: int,
    random_event_count: int,
) -> pd.DataFrame:
    ordered = catalog.sort_values(["anchor_index", "event_id"]).reset_index(drop=True)
    selections = [ordered.iloc[0].to_dict(), ordered.iloc[-1].to_dict()]
    selections[0]["selection_reason"] = "first_event"
    selections[1]["selection_reason"] = "last_event"
    candidates = ordered.iloc[1:-1]
    sample_size = min(random_event_count, len(candidates))
    if sample_size:
        rng = np.random.default_rng(random_seed)
        positions = sorted(rng.choice(len(candidates), size=sample_size, replace=False))
        for number, position in enumerate(positions, start=1):
            row = candidates.iloc[position].to_dict()
            row["selection_reason"] = f"random_{number}"
            selections.append(row)
    return pd.DataFrame(selections).drop_duplicates("event_id").reset_index(drop=True)


def event_rms_frame(event_frame: pd.DataFrame) -> pd.DataFrame:
    output = event_frame[["event_id", "relative_observation"]].copy()
    output["rms_16"] = trailing_rms(
        event_frame["original"].astype(float).to_numpy(), PRIMARY_WINDOW
    )
    return output


def build_pre_event_volatility_table(
    catalog: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    catalog_by_id = catalog.set_index("event_id")
    for event_id, event_frame in windows.groupby("event_id", sort=True):
        event_frame = event_frame.sort_values("relative_observation")
        rms = event_rms_frame(event_frame)
        pre = rms.loc[rms["relative_observation"] < 0, "rms_16"].dropna()
        trigger = rms.loc[rms["relative_observation"] == 0, "rms_16"]
        if pre.empty or len(trigger) != 1:
            raise ValueError(f"Event {event_id} lacks pre-event or trigger RMS")
        event = catalog_by_id.loc[event_id]
        rows.append({
            "event_id": int(event_id),
            "event_timestamp_utc": event["event_timestamp_utc"],
            "mean_pre_event_rms_16": float(pre.mean()),
            "trigger_rms_16": float(trigger.iloc[0]),
            "pre_event_rms_observation_count": int(len(pre)),
            "is_overlapping": bool(event["is_overlapping"]),
            "overlap_cluster_id": event["overlap_cluster_id"],
        })
    return pd.DataFrame(rows)


def plot_weekday_hour_counts(counts: pd.DataFrame, output_path: Path) -> Path:
    matrix = counts.pivot(index="weekday_number", columns="utc_hour", values="event_count").to_numpy()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(WEEKDAY_ORDER)
    ax.set_xlabel("UTC hour of event anchor")
    ax.set_ylabel("UTC weekday of event anchor")
    ax.set_title("Retained volatility events by UTC weekday and hour")
    for row in range(7):
        for column in range(24):
            value = int(matrix[row, column])
            ax.text(column, row, str(value), ha="center", va="center", fontsize=7,
                    color="white" if value > matrix.max() * 0.55 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Retained event count")
    fig.text(0.5, 0.01,
             "UTC day/hour use t₀, the end timestamp of the triggering 5-minute return interval.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_annual_event_counts(annual: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(annual["year"].astype(str), annual["event_count"], color=FINAL_COLOR)
    ax.set_xlabel("UTC year of event anchor")
    ax.set_ylabel("Retained event count")
    ax.set_title("Retained volatility events by year")
    ax.grid(axis="y", alpha=0.25)
    for index, count in enumerate(annual["event_count"]):
        ax.text(index, count, str(int(count)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_consecutive_event_intervals(intervals: pd.DataFrame, output_path: Path) -> Path:
    if intervals.empty:
        raise ValueError("At least two events are required for an interval histogram")
    values = intervals["interval_indexed_days"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(values, bins="auto", color=FINAL_COLOR, edgecolor="white", linewidth=0.7)
    median = float(np.median(values))
    ax.axvline(median, color="#b23a32", linestyle="--", linewidth=1.2,
               label=f"Median: {median:.2f} indexed days")
    ax.set_xlabel("Observations between consecutive events / 288")
    ax.set_ylabel("Consecutive event-pair count")
    ax.set_title(f"Time between consecutive retained events (n={len(values)})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def _relative_days(event_frame: pd.DataFrame) -> np.ndarray:
    return event_frame["relative_observation"].to_numpy() / OBSERVATIONS_PER_INDEXED_DAY


def plot_event_decomposition(event_frame: pd.DataFrame, output_path: Path, title: str) -> Path:
    layers = [("original", "Original"), *[(f"D_{i:02d}", f"D_{i:02d}") for i in range(1, 10)], ("A_09", "A_09")]
    x = _relative_days(event_frame)
    fig, axes = plt.subplots(len(layers), 1, figsize=(14, 15.5), sharex=True, constrained_layout=True)
    for axis, (column, label) in zip(axes, layers, strict=True):
        axis.plot(x, event_frame[column], color=FINAL_COLOR, linewidth=0.45, alpha=0.85, rasterized=True)
        axis.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        axis.axvline(0, color="#b23a32", linewidth=0.9, linestyle="--")
        axis.set_ylabel(label)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[-1].set_xlabel("Relative indexed trading days (t₀ = 0)")
    fig.suptitle(title)
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_event_rms(event_frame: pd.DataFrame, output_path: Path, title: str) -> Path:
    rms = event_rms_frame(event_frame)
    x = rms["relative_observation"].to_numpy() / OBSERVATIONS_PER_INDEXED_DAY
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, rms["rms_16"], color=FINAL_DARK_COLOR, linewidth=1.0)
    ax.axvline(0, color="#b23a32", linewidth=1.1, linestyle="--", label="Event anchor t₀")
    ax.set_xlabel("Relative indexed trading days")
    ax.set_ylabel("Trailing 80-minute RMS")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_pre_event_scatter(scatter: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for overlapping, rows in scatter.groupby("is_overlapping", sort=True):
        ax.scatter(rows["mean_pre_event_rms_16"], rows["trigger_rms_16"],
                   s=28, alpha=0.72, marker="o" if not overlapping else "x",
                   label="Overlapping event" if overlapping else "Non-overlapping event")
    x_max = float(scatter["mean_pre_event_rms_16"].max()) * 1.08
    y_max = float(scatter["trigger_rms_16"].max()) * 1.08
    identity_limit = min(x_max, y_max)
    ax.plot([0, identity_limit], [0, identity_limit], color="black", linestyle="--",
            linewidth=0.9, label="Identity")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Mean pre-event trailing 80-minute RMS")
    ax.set_ylabel("Trailing 80-minute RMS at t₀")
    ax.set_title("Pre-event average volatility versus trigger volatility")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path
