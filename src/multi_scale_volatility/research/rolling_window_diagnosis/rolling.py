"""Rolling window construction and decomposition diagnostics for V2.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.core.components import component_specs
from multi_scale_volatility.core.config.names import (
    BASE_INTERVAL_MINUTES,
    COMPONENT,
    COMPONENT_TYPE,
    DETAIL_ENERGY_SHARE,
    ENERGY,
    K,
    LOG_RETURN,
    RMS_VOLATILITY,
    SCALE_DAYS,
    SCALE_MINUTES,
    TIMESTAMP_UTC,
    TOTAL_COMPONENT_ENERGY_SHARE,
)
from multi_scale_volatility.core.config.paths import (
    FINAL_RETURNS_CSV,
    ROLLING_LAYER_VOLATILITY_CSV,
    ROLLING_REPORT_JSON,
    ROLLING_RESULTS_DIR,
    ROLLING_SCALE_GROUP_SUMMARY_CSV,
    ROLLING_WINDOW_METADATA_CSV,
    ROLLING_WINDOW_SUMMARY_CSV,
)
from multi_scale_volatility.core.config.schemas import RETURN_COLUMNS
from multi_scale_volatility.research.decomposition import (
    RECONSTRUCTION_TOLERANCE,
    decompose_values,
)
from multi_scale_volatility.core.io import write_csv, write_json
from multi_scale_volatility.core.utils.validation import require_finite_array, require_positive_k

ROLLING_WINDOW_LENGTHS = (2048, 8192)
ROLLING_STEP_SIZE = 288
ROLLING_K = 9
ROLLING_RANDOM_SEED = 20260624
ROLLING_SCALE_GROUPS = {
    "fine": ("D_01", "D_02", "D_03"),
    "mid": ("D_04", "D_05", "D_06"),
    "coarse": ("D_07", "D_08", "D_09"),
}
GROUP_SHARE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RollingPaths:
    input_csv: Path = FINAL_RETURNS_CSV
    output_dir: Path = ROLLING_RESULTS_DIR

    @property
    def metadata_csv(self) -> Path:
        return self.output_dir / ROLLING_WINDOW_METADATA_CSV.name

    @property
    def layer_volatility_csv(self) -> Path:
        return self.output_dir / ROLLING_LAYER_VOLATILITY_CSV.name

    @property
    def summary_csv(self) -> Path:
        return self.output_dir / ROLLING_WINDOW_SUMMARY_CSV.name

    @property
    def scale_group_summary_csv(self) -> Path:
        return self.output_dir / ROLLING_SCALE_GROUP_SUMMARY_CSV.name

    @property
    def report_json(self) -> Path:
        return self.output_dir / ROLLING_REPORT_JSON.name


@dataclass(frozen=True)
class RollingWindowSpec:
    window_length: int
    window_id: int
    start_index: int
    end_index: int


def compute_rolling_decomposition_diagnostics(
    paths: RollingPaths | None = None,
    window_lengths: tuple[int, ...] = ROLLING_WINDOW_LENGTHS,
    step_size: int = ROLLING_STEP_SIZE,
    k: int = ROLLING_K,
) -> dict[str, Any]:
    """Create rolling windows, validate decomposition, and save summary artifacts."""

    paths = paths or RollingPaths()
    require_positive_k(k)
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    for window_length in window_lengths:
        if window_length <= 0:
            raise ValueError("window lengths must be positive")
        if window_length % (2**k) != 0:
            raise ValueError(
                f"Window length {window_length} is not divisible by 2**{k}"
            )

    frame = pd.read_csv(paths.input_csv, usecols=list(RETURN_COLUMNS))
    if frame.empty:
        raise ValueError(f"Input dataset is empty: {paths.input_csv}")
    if frame[LOG_RETURN].isna().any():
        raise ValueError(f"Input contains NaN {LOG_RETURN} values: {paths.input_csv}")

    timestamps = frame[TIMESTAMP_UTC].astype(str).to_numpy()
    values = frame[LOG_RETURN].astype(float).to_numpy()
    require_finite_array(values, f"Input {LOG_RETURN} values in {paths.input_csv}")

    metadata_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    specs = component_specs(k, include_original=False, base_interval_minutes=BASE_INTERVAL_MINUTES)
    for window_length in window_lengths:
        for spec in rolling_window_specs(len(values), window_length, step_size):
            window_values = values[spec.start_index : spec.end_index + 1]
            diagnostics = decompose_window_values(window_values, k=k)
            metadata_rows.append(
                {
                    "window_length": spec.window_length,
                    "window_id": spec.window_id,
                    "window_start_index": spec.start_index,
                    "window_end_index": spec.end_index,
                    "window_start_timestamp_utc": timestamps[spec.start_index],
                    "window_end_timestamp_utc": timestamps[spec.end_index],
                    "step_size": step_size,
                    "K_roll": k,
                    "n_obs": window_length,
                }
            )
            summary_rows.append(
                {
                    "window_length": spec.window_length,
                    "window_id": spec.window_id,
                    "window_start_timestamp_utc": timestamps[spec.start_index],
                    "window_end_timestamp_utc": timestamps[spec.end_index],
                    "original_energy": diagnostics["original_energy"],
                    "original_rms_volatility": diagnostics["original_rms_volatility"],
                    "detail_energy_total": diagnostics["detail_energy_total"],
                    "approximation_energy": diagnostics["approximation_energy"],
                    "total_component_energy": diagnostics["total_component_energy"],
                    "energy_reconstruction_gap": diagnostics["energy_reconstruction_gap"],
                    "max_abs_reconstruction_error": diagnostics[
                        "max_abs_reconstruction_error"
                    ],
                    "mean_abs_reconstruction_error": diagnostics[
                        "mean_abs_reconstruction_error"
                    ],
                }
            )
            layer_rows.extend(
                rolling_layer_volatility_rows(
                    spec,
                    timestamps[spec.end_index],
                    diagnostics,
                    specs=specs,
                )
            )
            group_rows.extend(
                rolling_scale_group_rows(
                    spec,
                    timestamps[spec.end_index],
                    diagnostics,
                )
            )

    metadata = pd.DataFrame(metadata_rows)
    layer_volatility = pd.DataFrame(layer_rows)
    summary = pd.DataFrame(summary_rows)
    scale_group_summary = pd.DataFrame(group_rows)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metadata, paths.metadata_csv, index=False)
    write_csv(layer_volatility, paths.layer_volatility_csv, index=False)
    write_csv(summary, paths.summary_csv, index=False)
    write_csv(scale_group_summary, paths.scale_group_summary_csv, index=False)

    report = {
        "input_csv": str(paths.input_csv),
        "metadata_csv": str(paths.metadata_csv),
        "layer_volatility_csv": str(paths.layer_volatility_csv),
        "summary_csv": str(paths.summary_csv),
        "scale_group_summary_csv": str(paths.scale_group_summary_csv),
        "N": int(len(values)),
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "window_lengths": list(window_lengths),
        "step_size": int(step_size),
        "K_roll": int(k),
        "block_size_max": int(2**k),
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "window_counts": {
            str(window_length): int(
                (metadata["window_length"] == window_length).sum()
            )
            for window_length in window_lengths
        },
        "max_abs_reconstruction_error": float(
            summary["max_abs_reconstruction_error"].max()
        ),
        "max_energy_reconstruction_gap_abs": float(
            summary["energy_reconstruction_gap"].abs().max()
        ),
        "max_group_detail_energy_share_sum_gap_abs": float(
            scale_group_summary.groupby(["window_length", "window_id"])[
                "group_detail_energy_share"
            ].sum().sub(1.0).abs().max()
        ),
    }
    write_json(paths.report_json, report)
    return report


def rolling_window_specs(
    n: int,
    window_length: int,
    step_size: int,
) -> list[RollingWindowSpec]:
    if n < window_length:
        return []
    return [
        RollingWindowSpec(
            window_length=window_length,
            window_id=window_id,
            start_index=start_index,
            end_index=start_index + window_length - 1,
        )
        for window_id, start_index in enumerate(range(0, n - window_length + 1, step_size))
    ]


def decompose_window_values(values: np.ndarray, k: int = ROLLING_K) -> dict[str, Any]:
    details, approximation = decompose_values(values, k=k)
    reconstruction = approximation.copy()
    for detail in details:
        reconstruction += detail

    error = values - reconstruction
    max_abs_error = float(np.max(np.abs(error)))
    mean_abs_error = float(np.mean(np.abs(error)))
    if max_abs_error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"Rolling reconstruction error {max_abs_error} exceeds "
            f"{RECONSTRUCTION_TOLERANCE}"
        )

    original_energy = float(np.dot(values, values))
    detail_energy_total = float(sum(float(np.dot(detail, detail)) for detail in details))
    approximation_energy = float(np.dot(approximation, approximation))
    total_component_energy = detail_energy_total + approximation_energy
    return {
        "details": details,
        "approximation": approximation,
        "component_energies": {
            **{
                f"D_{scale:02d}": float(np.dot(detail, detail))
                for scale, detail in enumerate(details, start=1)
            },
            f"A_{k:02d}": approximation_energy,
        },
        "original_energy": original_energy,
        "original_rms_volatility": float(np.sqrt(original_energy / len(values))),
        "detail_energy_total": detail_energy_total,
        "approximation_energy": approximation_energy,
        "total_component_energy": total_component_energy,
        "energy_reconstruction_gap": original_energy - total_component_energy,
        "max_abs_reconstruction_error": max_abs_error,
        "mean_abs_reconstruction_error": mean_abs_error,
    }


def rolling_layer_volatility_rows(
    spec: RollingWindowSpec,
    window_end_timestamp_utc: str,
    diagnostics: dict[str, Any],
    specs: list[Any],
) -> list[dict[str, Any]]:
    component_energies = diagnostics["component_energies"]
    detail_energy_total = diagnostics["detail_energy_total"]
    total_component_energy = diagnostics["total_component_energy"]
    rows: list[dict[str, Any]] = []
    for component_spec in specs:
        component = component_spec.name
        energy = float(component_energies[component])
        detail_share = np.nan
        if component_spec.kind == "detail":
            detail_share = energy / detail_energy_total
        rows.append(
            {
                "window_length": spec.window_length,
                "window_id": spec.window_id,
                "window_end_timestamp_utc": window_end_timestamp_utc,
                COMPONENT: component,
                K: component_spec.scale,
                COMPONENT_TYPE: component_spec.kind,
                SCALE_MINUTES: component_spec.scale_minutes,
                SCALE_DAYS: component_spec.scale_days,
                ENERGY: energy,
                RMS_VOLATILITY: float(np.sqrt(energy / spec.window_length)),
                DETAIL_ENERGY_SHARE: detail_share,
                TOTAL_COMPONENT_ENERGY_SHARE: energy / total_component_energy,
            }
        )
    return rows


def rolling_scale_group_rows(
    spec: RollingWindowSpec,
    window_end_timestamp_utc: str,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    component_energies = diagnostics["component_energies"]
    detail_energy_total = diagnostics["detail_energy_total"]
    rows: list[dict[str, Any]] = []
    for group_name, components in ROLLING_SCALE_GROUPS.items():
        group_energy = float(sum(component_energies[component] for component in components))
        rows.append(
            {
                "window_length": spec.window_length,
                "window_id": spec.window_id,
                "window_end_timestamp_utc": window_end_timestamp_utc,
                "scale_group": group_name,
                "component_start": components[0],
                "component_end": components[-1],
                "group_energy": group_energy,
                "group_detail_energy_share": group_energy / detail_energy_total,
            }
        )
    group_share_sum = sum(row["group_detail_energy_share"] for row in rows)
    if abs(group_share_sum - 1.0) > GROUP_SHARE_TOLERANCE:
        raise ValueError(
            f"Scale group shares sum to {group_share_sum} for "
            f"W={spec.window_length}, window_id={spec.window_id}"
        )
    return rows


def decompose_rolling_window_from_input(
    input_csv: Path,
    window_length: int,
    window_id: int,
    step_size: int = ROLLING_STEP_SIZE,
    k: int = ROLLING_K,
) -> pd.DataFrame:
    frame = pd.read_csv(input_csv, usecols=list(RETURN_COLUMNS))
    values = frame[LOG_RETURN].astype(float).to_numpy()
    timestamps = frame[TIMESTAMP_UTC].astype(str).to_numpy()
    specs = rolling_window_specs(len(values), window_length, step_size)
    if window_id < 0 or window_id >= len(specs):
        raise ValueError(
            f"window_id {window_id} is outside valid range 0..{len(specs) - 1}"
        )
    spec = specs[window_id]
    window_values = values[spec.start_index : spec.end_index + 1]
    diagnostics = decompose_window_values(window_values, k=k)

    output = pd.DataFrame(
        {
            "index": np.arange(len(window_values), dtype=np.int64),
            "source_index": np.arange(spec.start_index, spec.end_index + 1, dtype=np.int64),
            TIMESTAMP_UTC: timestamps[spec.start_index : spec.end_index + 1],
            "original": window_values,
        }
    )
    for scale, detail in enumerate(diagnostics["details"], start=1):
        output[f"D_{scale:02d}"] = detail
    output[f"A_{k:02d}"] = diagnostics["approximation"]
    return output
