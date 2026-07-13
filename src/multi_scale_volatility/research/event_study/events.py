"""Causal volatility-event detection and event-aligned decomposition for V3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from multi_scale_volatility.app.runtime import get_logger
from multi_scale_volatility.core.components import component_specs
from multi_scale_volatility.core.config.names import LOG_RETURN, TIMESTAMP_UTC
from multi_scale_volatility.core.config.paths import (
    EVENT_CATALOG_CSV,
    EVENT_COMPONENT_SUMMARY_CSV,
    EVENT_DETECTION_REPORT_JSON,
    EVENT_DETECTION_SERIES_PARQUET,
    EVENT_SCALE_GROUP_SUMMARY_CSV,
    EVENT_STUDY_RESULTS_DIR,
    EVENT_WINDOWS_PARQUET,
    EVENT_WINDOWS_REPORT_JSON,
    FINAL_RETURNS_CSV,
)
from multi_scale_volatility.core.io import write_csv, write_json, write_parquet
from multi_scale_volatility.core.utils.validation import require_finite_array
from multi_scale_volatility.research.decomposition import RECONSTRUCTION_TOLERANCE, decompose_values
from multi_scale_volatility.research.rolling_window_diagnosis.rolling import (
    ROLLING_SCALE_GROUPS,
)

PRIMARY_WINDOW = 16
REFERENCE_LENGTH = 5760
PRIMARY_THRESHOLD = 3.0
RESET_THRESHOLD = 1.0
RESET_LENGTH = 12
PRE_EVENT_OBSERVATIONS = 1440
POST_EVENT_END_OFFSET = 3167
EVENT_WINDOW_LENGTH = 4608
EVENT_K = 9
MAD_NORMALIZATION = 1.4826
REFERENCE_CHUNK_SIZE = 512

logger = get_logger(__name__)


@dataclass(frozen=True)
class EventStudyPaths:
    input_csv: Path = FINAL_RETURNS_CSV
    output_dir: Path = EVENT_STUDY_RESULTS_DIR

    @property
    def detection_series_parquet(self) -> Path:
        return self.output_dir / EVENT_DETECTION_SERIES_PARQUET.name

    @property
    def event_catalog_csv(self) -> Path:
        return self.output_dir / EVENT_CATALOG_CSV.name

    @property
    def detection_report_json(self) -> Path:
        return self.output_dir / EVENT_DETECTION_REPORT_JSON.name

    @property
    def event_windows_parquet(self) -> Path:
        return self.output_dir / EVENT_WINDOWS_PARQUET.name

    @property
    def component_summary_csv(self) -> Path:
        return self.output_dir / EVENT_COMPONENT_SUMMARY_CSV.name

    @property
    def scale_group_summary_csv(self) -> Path:
        return self.output_dir / EVENT_SCALE_GROUP_SUMMARY_CSV.name

    @property
    def windows_report_json(self) -> Path:
        return self.output_dir / EVENT_WINDOWS_REPORT_JSON.name


def trailing_rms(values: np.ndarray, window: int) -> np.ndarray:
    """Return trailing RMS including the current observation."""
    if window <= 0:
        raise ValueError("window must be positive")
    result = np.full(len(values), np.nan)
    if len(values) < window:
        return result
    squared = np.square(values, dtype=float)
    sums = np.cumsum(np.insert(squared, 0, 0.0))
    result[window - 1 :] = np.sqrt((sums[window:] - sums[:-window]) / window)
    return result


def robust_causal_scores(
    volatility: np.ndarray,
    reference_length: int = REFERENCE_LENGTH,
    chunk_size: int = REFERENCE_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score log volatility against the exact preceding-window median and MAD."""
    if reference_length <= 0 or chunk_size <= 0:
        raise ValueError("reference_length and chunk_size must be positive")
    n = len(volatility)
    medians = np.full(n, np.nan)
    mads = np.full(n, np.nan)
    scores = np.full(n, np.nan)
    finite_start = int(np.argmax(np.isfinite(volatility))) if np.isfinite(volatility).any() else n
    if finite_start == n:
        return medians, mads, scores
    logs = np.full(n, np.nan)
    positive = volatility > 0
    logs[positive] = np.log(volatility[positive])
    valid_logs = logs[finite_start:]
    if len(valid_logs) <= reference_length:
        return medians, mads, scores
    windows = sliding_window_view(valid_logs, reference_length)
    target_start = finite_start + reference_length
    reference_windows = windows[:-1]
    for start in range(0, len(reference_windows), chunk_size):
        stop = min(start + chunk_size, len(reference_windows))
        block = reference_windows[start:stop]
        complete = np.isfinite(block).all(axis=1)
        block_medians = np.full(len(block), np.nan)
        block_mads = np.full(len(block), np.nan)
        if complete.any():
            complete_block = block[complete]
            complete_medians = np.median(complete_block, axis=1)
            block_medians[complete] = complete_medians
            block_mads[complete] = np.median(
                np.abs(complete_block - complete_medians[:, None]), axis=1
            )
        output = slice(target_start + start, target_start + stop)
        medians[output] = block_medians
        mads[output] = block_mads
    denominator = MAD_NORMALIZATION * mads
    usable = np.isfinite(logs) & (denominator > 0)
    scores[usable] = (logs[usable] - medians[usable]) / denominator[usable]
    return medians, mads, scores


def apply_state_machine(
    primary_scores: np.ndarray,
    primary_threshold: float = PRIMARY_THRESHOLD,
    reset_threshold: float = RESET_THRESHOLD,
    reset_length: int = RESET_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply crossing-trigger and consecutive-observation reset rules."""
    if reset_length <= 0:
        raise ValueError("reset_length must be positive")
    n = len(primary_scores)
    triggers = np.zeros(n, dtype=bool)
    resets = np.zeros(n, dtype=bool)
    active = np.zeros(n, dtype=bool)
    streaks = np.zeros(n, dtype=np.int64)
    ready = True
    streak = 0
    for index in range(1, n):
        if ready:
            crossing = (
                np.isfinite(primary_scores[index - 1])
                and np.isfinite(primary_scores[index])
                and primary_scores[index - 1] < primary_threshold
                and primary_scores[index] >= primary_threshold
            )
            if crossing:
                triggers[index] = True
                ready = False
                streak = 0
        else:
            below_reset = (
                np.isfinite(primary_scores[index])
                and primary_scores[index] < reset_threshold
            )
            streak = streak + 1 if below_reset else 0
            if streak >= reset_length:
                resets[index] = True
                ready = True
                streak = 0
        active[index] = not ready
        streaks[index] = streak
    return triggers, resets, active, streaks


def annotate_overlaps(catalog: pd.DataFrame) -> pd.DataFrame:
    """Annotate direct overlaps and transitive overlap clusters."""
    output = catalog.copy()
    n = len(output)
    output["overlaps_previous"] = False
    output["overlaps_next"] = False
    output["overlap_event_count"] = 0
    output["overlap_cluster_id"] = pd.Series([pd.NA] * n, dtype="Int64")
    output["is_overlapping"] = False
    eligible_positions = [i for i in range(n) if bool(output.iloc[i]["is_window_eligible"])]
    cluster_id = -1
    cluster: list[int] = []
    cluster_end = -1

    def finish_cluster() -> None:
        nonlocal cluster_id, cluster
        if len(cluster) < 2:
            cluster = []
            return
        cluster_id += 1
        for position in cluster:
            start = int(output.iloc[position]["window_start_index"])
            end = int(output.iloc[position]["window_end_index"])
            direct = 0
            for other in cluster:
                if other == position:
                    continue
                other_start = int(output.iloc[other]["window_start_index"])
                other_end = int(output.iloc[other]["window_end_index"])
                direct += int(start <= other_end and other_start <= end)
            output.at[position, "overlap_event_count"] = direct
            output.at[position, "overlap_cluster_id"] = cluster_id
            output.at[position, "is_overlapping"] = True
        cluster = []

    for position in eligible_positions:
        start = int(output.iloc[position]["window_start_index"])
        end = int(output.iloc[position]["window_end_index"])
        if cluster and start > cluster_end:
            finish_cluster()
            cluster_end = -1
        cluster.append(position)
        cluster_end = max(cluster_end, end)
    finish_cluster()

    for left, right in zip(eligible_positions, eligible_positions[1:]):
        overlaps = int(output.iloc[right]["window_start_index"]) <= int(
            output.iloc[left]["window_end_index"]
        )
        if overlaps:
            output.at[left, "overlaps_next"] = True
            output.at[right, "overlaps_previous"] = True
    return output


def build_event_catalog(
    timestamps: np.ndarray,
    trigger_indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(timestamps)
    for event_id, anchor in enumerate(trigger_indices):
        start = int(anchor) - PRE_EVENT_OBSERVATIONS
        end = int(anchor) + POST_EVENT_END_OFFSET
        eligible = start >= 0 and end < n
        rows.append(
            {
                "event_id": event_id,
                "anchor_index": int(anchor),
                "event_timestamp_utc": timestamps[anchor],
                "window_start_index": start,
                "window_end_index": end,
                "window_start_timestamp_utc": timestamps[start] if start >= 0 else pd.NA,
                "window_end_timestamp_utc": timestamps[end] if end < n else pd.NA,
                "is_window_eligible": eligible,
            }
        )
    columns = [
        "event_id", "anchor_index", "event_timestamp_utc", "window_start_index",
        "window_end_index", "window_start_timestamp_utc", "window_end_timestamp_utc",
        "is_window_eligible",
    ]
    return annotate_overlaps(pd.DataFrame(rows, columns=columns))


def detect_events(
    paths: EventStudyPaths | None = None,
    reference_length: int = REFERENCE_LENGTH,
    chunk_size: int = REFERENCE_CHUNK_SIZE,
) -> dict[str, Any]:
    """Detect events and write the observation-level audit and event catalog."""
    paths = paths or EventStudyPaths()
    frame = pd.read_csv(paths.input_csv, usecols=[TIMESTAMP_UTC, LOG_RETURN])
    if frame.empty:
        raise ValueError(f"Input dataset is empty: {paths.input_csv}")
    values = frame[LOG_RETURN].astype(float).to_numpy()
    require_finite_array(values, LOG_RETURN)
    timestamps = frame[TIMESTAMP_UTC].astype(str).to_numpy()
    volatility = trailing_rms(values, PRIMARY_WINDOW)
    median, mad, score = robust_causal_scores(volatility, reference_length, chunk_size)
    triggers, resets, active, streaks = apply_state_machine(score)
    audit = pd.DataFrame({
        "source_index": np.arange(len(frame), dtype=np.int64),
        TIMESTAMP_UTC: timestamps,
        LOG_RETURN: values,
        "rms_16": volatility,
        "reference_median_log_rms_16": median,
        "reference_mad_log_rms_16": mad,
        "robust_score_16": score,
        "event_trigger": triggers,
        "reset_completed": resets,
        "detector_active": active,
        "reset_streak": streaks,
    })
    catalog = build_event_catalog(timestamps, np.flatnonzero(triggers))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(audit, paths.detection_series_parquet, index=False)
    write_csv(catalog, paths.event_catalog_csv, index=False)
    report = {
        "input_csv": str(paths.input_csv),
        "detection_series_parquet": str(paths.detection_series_parquet),
        "event_catalog_csv": str(paths.event_catalog_csv),
        "N": len(frame),
        "event_count": int(triggers.sum()),
        "eligible_event_count": int(catalog["is_window_eligible"].sum()),
        "overlapping_event_count": int(catalog["is_overlapping"].sum()),
        "overlap_cluster_count": int(catalog["overlap_cluster_id"].nunique()),
        "primary_window": PRIMARY_WINDOW,
        "reference_length": reference_length,
        "primary_threshold": PRIMARY_THRESHOLD,
        "reset_threshold": RESET_THRESHOLD,
        "reset_length": RESET_LENGTH,
        "window_relative_offsets_inclusive": [-PRE_EVENT_OBSERVATIONS, POST_EVENT_END_OFFSET],
        "event_window_length": EVENT_WINDOW_LENGTH,
        "undefined_score_count": int(np.isnan(score).sum()),
        "zero_mad_count": int((mad == 0).sum()),
    }
    write_json(paths.detection_report_json, report)
    return report


def extract_event_windows(
    paths: EventStudyPaths | None = None,
    k: int = EVENT_K,
) -> dict[str, Any]:
    """Extract and decompose all eligible fixed event windows."""
    paths = paths or EventStudyPaths()
    frame = pd.read_csv(paths.input_csv, usecols=[TIMESTAMP_UTC, LOG_RETURN])
    catalog = pd.read_csv(paths.event_catalog_csv)
    values = frame[LOG_RETURN].astype(float).to_numpy()
    timestamps = frame[TIMESTAMP_UTC].astype(str).to_numpy()
    if EVENT_WINDOW_LENGTH % (2**k) != 0:
        raise ValueError(f"Event window length {EVENT_WINDOW_LENGTH} is not divisible by 2**{k}")
    window_frames: list[pd.DataFrame] = []
    component_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    max_error = 0.0
    for event in catalog.itertuples(index=False):
        if not bool(event.is_window_eligible):
            continue
        start, end = int(event.window_start_index), int(event.window_end_index)
        event_values = values[start : end + 1]
        if len(event_values) != EVENT_WINDOW_LENGTH:
            raise ValueError(f"Event {event.event_id} has {len(event_values)} observations")
        details, approximation = decompose_values(event_values, k=k)
        reconstruction = approximation.copy()
        for detail in details:
            reconstruction += detail
        error = float(np.max(np.abs(event_values - reconstruction)))
        if error > RECONSTRUCTION_TOLERANCE:
            raise ValueError(f"Event {event.event_id} reconstruction error {error}")
        max_error = max(max_error, error)
        components = {f"D_{i:02d}": detail for i, detail in enumerate(details, 1)}
        components[f"A_{k:02d}"] = approximation
        window = pd.DataFrame({
            "event_id": int(event.event_id),
            "relative_observation": np.arange(-PRE_EVENT_OBSERVATIONS, POST_EVENT_END_OFFSET + 1),
            "source_index": np.arange(start, end + 1),
            TIMESTAMP_UTC: timestamps[start : end + 1],
            "original": event_values,
            **components,
        })
        window_frames.append(window)
        energies = {name: float(np.dot(data, data)) for name, data in components.items()}
        detail_total = sum(energies[f"D_{i:02d}"] for i in range(1, k + 1))
        total = sum(energies.values())
        for spec in component_specs(k):
            energy = energies[spec.name]
            component_rows.append({
                "event_id": int(event.event_id), "component": spec.name,
                "component_type": spec.kind, "k": spec.scale,
                "scale_minutes": spec.scale_minutes, "scale_days": spec.scale_days,
                "energy": energy, "rms_volatility": np.sqrt(energy / EVENT_WINDOW_LENGTH),
                "detail_energy_share": energy / detail_total if spec.kind == "detail" else np.nan,
                "total_component_energy_share": energy / total,
            })
        for group, names in ROLLING_SCALE_GROUPS.items():
            energy = sum(energies[name] for name in names)
            group_rows.append({
                "event_id": int(event.event_id), "scale_group": group,
                "component_start": names[0], "component_end": names[-1],
                "group_energy": energy, "group_rms_volatility": np.sqrt(energy / EVENT_WINDOW_LENGTH),
                "group_detail_energy_share": energy / detail_total,
            })
    windows = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
    write_parquet(windows, paths.event_windows_parquet, index=False)
    write_csv(pd.DataFrame(component_rows), paths.component_summary_csv, index=False)
    write_csv(pd.DataFrame(group_rows), paths.scale_group_summary_csv, index=False)
    report = {
        "input_csv": str(paths.input_csv), "event_catalog_csv": str(paths.event_catalog_csv),
        "event_windows_parquet": str(paths.event_windows_parquet),
        "component_summary_csv": str(paths.component_summary_csv),
        "scale_group_summary_csv": str(paths.scale_group_summary_csv),
        "eligible_event_count": int(catalog["is_window_eligible"].sum()),
        "event_window_length": EVENT_WINDOW_LENGTH, "K": k,
        "block_size_max": 2**k, "max_abs_reconstruction_error": max_error,
    }
    write_json(paths.windows_report_json, report)
    return report
