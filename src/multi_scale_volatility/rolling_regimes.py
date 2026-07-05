"""Two-dimensional rolling volatility-state maps for V2.3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.config.paths import (
    ROLLING_REGIME_CELL_COUNTS_CSV,
    ROLLING_REGIME_EPISODE_COUNTS_CSV,
    ROLLING_REGIME_EPISODE_SUMMARY_CSV,
    ROLLING_REGIME_METRICS_CSV,
    ROLLING_REGIME_REPORT_JSON,
    ROLLING_REGIME_RESULTS_DIR,
    ROLLING_RESULTS_DIR,
    ROLLING_SCALE_GROUP_SUMMARY_CSV,
    ROLLING_WINDOW_SUMMARY_CSV,
)
from multi_scale_volatility.io import write_csv, write_json
from multi_scale_volatility.rolling import ROLLING_STEP_SIZE

VOL_BUCKET_ORDER = ("lowVol", "midVol", "highVol")
FINE_BUCKET_ORDER = ("lowFine", "midFine", "highFine")
PROFILE_REGIMES = (
    "highVol_highFine",
    "highVol_lowFine",
    "lowVol_highFine",
    "lowVol_lowFine",
)
MIN_EPISODE_WINDOWS = 3


@dataclass(frozen=True)
class RollingRegimePaths:
    results_dir: Path = ROLLING_RESULTS_DIR
    output_dir: Path = ROLLING_REGIME_RESULTS_DIR

    @property
    def summary_csv(self) -> Path:
        return self.results_dir / ROLLING_WINDOW_SUMMARY_CSV.name

    @property
    def scale_group_summary_csv(self) -> Path:
        return self.results_dir / ROLLING_SCALE_GROUP_SUMMARY_CSV.name

    @property
    def regime_metrics_csv(self) -> Path:
        return self.output_dir / ROLLING_REGIME_METRICS_CSV.name

    @property
    def episode_summary_csv(self) -> Path:
        return self.output_dir / ROLLING_REGIME_EPISODE_SUMMARY_CSV.name

    @property
    def cell_counts_csv(self) -> Path:
        return self.output_dir / ROLLING_REGIME_CELL_COUNTS_CSV.name

    @property
    def episode_counts_csv(self) -> Path:
        return self.output_dir / ROLLING_REGIME_EPISODE_COUNTS_CSV.name

    @property
    def report_json(self) -> Path:
        return self.output_dir / ROLLING_REGIME_REPORT_JSON.name


def compute_rolling_regime_diagnostics(
    paths: RollingRegimePaths | None = None,
    min_episode_windows: int = MIN_EPISODE_WINDOWS,
) -> dict[str, Any]:
    """Create V2.3 rolling regime metrics, counts, and episode summaries."""

    if min_episode_windows <= 0:
        raise ValueError("min_episode_windows must be positive")

    paths = paths or RollingRegimePaths()
    summary = pd.read_csv(paths.summary_csv)
    groups = pd.read_csv(paths.scale_group_summary_csv)

    metrics = build_regime_metrics(summary, groups)
    episodes = build_episode_summary(metrics, min_episode_windows=min_episode_windows)
    cell_counts = build_cell_counts(metrics)
    episode_counts = build_episode_counts(episodes)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics, paths.regime_metrics_csv, index=False)
    write_csv(episodes, paths.episode_summary_csv, index=False)
    write_csv(cell_counts, paths.cell_counts_csv, index=False)
    write_csv(episode_counts, paths.episode_counts_csv, index=False)

    report = {
        "summary_csv": str(paths.summary_csv),
        "scale_group_summary_csv": str(paths.scale_group_summary_csv),
        "regime_metrics_csv": str(paths.regime_metrics_csv),
        "episode_summary_csv": str(paths.episode_summary_csv),
        "cell_counts_csv": str(paths.cell_counts_csv),
        "episode_counts_csv": str(paths.episode_counts_csv),
        "window_counts": {
            str(window_length): int(count)
            for window_length, count in metrics.groupby("window_length").size().items()
        },
        "episode_counts_after_filter": {
            str(window_length): int(count)
            for window_length, count in episodes.groupby("window_length").size().items()
        },
        "percentile_convention": 'rank(method="average", pct=True) within window_length',
        "vol_bucket_thresholds": {"low_max": 0.20, "high_min": 0.80},
        "fine_bucket_thresholds": {"low_max": 0.20, "high_min": 0.80},
        "min_episode_windows": int(min_episode_windows),
        "step_size_observations": ROLLING_STEP_SIZE,
        "profile_regimes": list(PROFILE_REGIMES),
    }
    write_json(paths.report_json, report)
    return report


def build_regime_metrics(summary: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    required_summary = {
        "window_length",
        "window_id",
        "window_start_timestamp_utc",
        "window_end_timestamp_utc",
        "original_rms_volatility",
    }
    required_groups = {
        "window_length",
        "window_id",
        "scale_group",
        "group_detail_energy_share",
    }
    missing_summary = required_summary.difference(summary.columns)
    missing_groups = required_groups.difference(groups.columns)
    if missing_summary:
        raise ValueError(f"Missing summary columns: {sorted(missing_summary)}")
    if missing_groups:
        raise ValueError(f"Missing scale group columns: {sorted(missing_groups)}")

    shares = (
        groups.pivot_table(
            index=["window_length", "window_id"],
            columns="scale_group",
            values="group_detail_energy_share",
            aggfunc="first",
        )
        .rename(
            columns={
                "fine": "fine_detail_energy_share",
                "mid": "mid_detail_energy_share",
                "coarse": "coarse_detail_energy_share",
            }
        )
        .reset_index()
    )
    needed_shares = {
        "fine_detail_energy_share",
        "mid_detail_energy_share",
        "coarse_detail_energy_share",
    }
    missing_shares = needed_shares.difference(shares.columns)
    if missing_shares:
        raise ValueError(f"Missing scale group shares: {sorted(missing_shares)}")

    metrics = summary[
        [
            "window_length",
            "window_id",
            "window_start_timestamp_utc",
            "window_end_timestamp_utc",
            "original_rms_volatility",
        ]
    ].merge(shares, on=["window_length", "window_id"], how="inner", validate="one_to_one")

    metrics["total_rms_percentile"] = metrics.groupby("window_length")[
        "original_rms_volatility"
    ].rank(method="average", pct=True)
    metrics["fine_share_percentile"] = metrics.groupby("window_length")[
        "fine_detail_energy_share"
    ].rank(method="average", pct=True)
    metrics["vol_bucket"] = metrics["total_rms_percentile"].map(bucket_volatility)
    metrics["fine_bucket"] = metrics["fine_share_percentile"].map(bucket_fine_share)
    metrics["regime_label"] = metrics["vol_bucket"] + "_" + metrics["fine_bucket"]
    metrics["year"] = pd.to_datetime(
        metrics["window_end_timestamp_utc"], utc=True
    ).dt.year
    return metrics.sort_values(["window_length", "window_id"]).reset_index(drop=True)


def bucket_volatility(value: float) -> str:
    if value <= 0.20:
        return "lowVol"
    if value >= 0.80:
        return "highVol"
    return "midVol"


def bucket_fine_share(value: float) -> str:
    if value <= 0.20:
        return "lowFine"
    if value >= 0.80:
        return "highFine"
    return "midFine"


def build_cell_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_length, group in metrics.groupby("window_length", sort=True):
        total = len(group)
        counts = group.groupby(["fine_bucket", "vol_bucket"]).size()
        for fine_bucket in FINE_BUCKET_ORDER:
            for vol_bucket in VOL_BUCKET_ORDER:
                count = int(counts.get((fine_bucket, vol_bucket), 0))
                rows.append(
                    {
                        "window_length": int(window_length),
                        "fine_bucket": fine_bucket,
                        "vol_bucket": vol_bucket,
                        "regime_label": f"{vol_bucket}_{fine_bucket}",
                        "window_count": count,
                        "window_share": count / total if total else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_episode_summary(
    metrics: pd.DataFrame,
    min_episode_windows: int = MIN_EPISODE_WINDOWS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    next_episode_id = 0
    for window_length, group in metrics.groupby("window_length", sort=True):
        ordered = group.sort_values("window_id").reset_index(drop=True)
        run_break = (
            (ordered["regime_label"] != ordered["regime_label"].shift())
            | (ordered["window_id"].diff().fillna(1) != 1)
        )
        ordered = ordered.assign(run_id=run_break.cumsum())
        for _, episode in ordered.groupby("run_id", sort=True):
            duration_windows = int(len(episode))
            if duration_windows < min_episode_windows:
                continue
            peak_index = episode["original_rms_volatility"].idxmax()
            peak = episode.loc[peak_index]
            rows.append(
                {
                    "window_length": int(window_length),
                    "episode_id": next_episode_id,
                    "regime_label": str(episode["regime_label"].iloc[0]),
                    "start_timestamp_utc": episode["window_start_timestamp_utc"].iloc[0],
                    "end_timestamp_utc": episode["window_end_timestamp_utc"].iloc[-1],
                    "start_window_id": int(episode["window_id"].iloc[0]),
                    "end_window_id": int(episode["window_id"].iloc[-1]),
                    "duration_windows": duration_windows,
                    "duration_trading_days": float(duration_windows),
                    "mean_total_rms": float(episode["original_rms_volatility"].mean()),
                    "max_total_rms": float(episode["original_rms_volatility"].max()),
                    "mean_total_rms_percentile": float(
                        episode["total_rms_percentile"].mean()
                    ),
                    "mean_fine_share": float(episode["fine_detail_energy_share"].mean()),
                    "mean_fine_share_percentile": float(
                        episode["fine_share_percentile"].mean()
                    ),
                    "mean_mid_share": float(episode["mid_detail_energy_share"].mean()),
                    "mean_coarse_share": float(
                        episode["coarse_detail_energy_share"].mean()
                    ),
                    "peak_window_timestamp_utc": peak["window_end_timestamp_utc"],
                }
            )
            next_episode_id += 1
    return pd.DataFrame(rows)


def build_episode_counts(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if episodes.empty:
        window_lengths: list[int] = []
    else:
        window_lengths = sorted(int(value) for value in episodes["window_length"].unique())
    for window_length in window_lengths:
        group = episodes[episodes["window_length"] == window_length]
        total = len(group)
        counts = group.groupby("regime_label").size()
        for fine_bucket in FINE_BUCKET_ORDER:
            for vol_bucket in VOL_BUCKET_ORDER:
                regime_label = f"{vol_bucket}_{fine_bucket}"
                count = int(counts.get(regime_label, 0))
                rows.append(
                    {
                        "window_length": int(window_length),
                        "fine_bucket": fine_bucket,
                        "vol_bucket": vol_bucket,
                        "regime_label": regime_label,
                        "episode_count": count,
                        "episode_share": count / total if total else np.nan,
                    }
                )
    return pd.DataFrame(rows)
