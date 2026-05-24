"""Volatility and energy metrics for decomposition components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.scale_utils import (
    component_scale,
    component_scale_minutes,
    component_type,
    decomposition_components,
)
from src.globals.columns import COMPONENT_TYPE, ORIGINAL, SERIES
from src.globals.constants import (
    BASE_INTERVAL_MINUTES,
    DEFAULT_K,
    PERIODS_PER_HOUR,
    TRADING_DAYS_PER_YEAR,
    TRADING_HOURS_PER_DAY,
)
from src.globals.paths import (
    FINAL_DECOMPOSITION_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    SHUFFLE_DECOMPOSITION_CSV,
    VOLATILITY_CSV,
    VOLATILITY_REPORT_JSON,
    VOLATILITY_RESULTS_DIR,
)
from src.globals.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_SHUFFLE
from src.utils.artifact_io import write_csv
from src.utils.json_utils import write_json
from src.utils.validation import require_finite_array, require_positive_k


@dataclass(frozen=True)
class VolatilityInput:
    name: str
    decomposition_csv: Path


@dataclass(frozen=True)
class VolatilityPaths:
    final_decomposition_csv: Path = FINAL_DECOMPOSITION_CSV
    shuffle_decomposition_csv: Path = SHUFFLE_DECOMPOSITION_CSV
    gaussian_decomposition_csv: Path = GAUSSIAN_DECOMPOSITION_CSV
    output_dir: Path = VOLATILITY_RESULTS_DIR

    @property
    def output_csv(self) -> Path:
        return VOLATILITY_CSV if self.output_dir == VOLATILITY_RESULTS_DIR else (
            self.output_dir / VOLATILITY_CSV.name
        )

    @property
    def report_json(self) -> Path:
        return VOLATILITY_REPORT_JSON if self.output_dir == VOLATILITY_RESULTS_DIR else (
            self.output_dir / VOLATILITY_REPORT_JSON.name
        )

    def inputs(self) -> list[VolatilityInput]:
        return [
            VolatilityInput(SERIES_FINAL, self.final_decomposition_csv),
            VolatilityInput(SERIES_SHUFFLE, self.shuffle_decomposition_csv),
            VolatilityInput(SERIES_GAUSSIAN, self.gaussian_decomposition_csv),
        ]


def compute_volatility_metrics(
    paths: VolatilityPaths | None = None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    paths = paths or VolatilityPaths()
    require_positive_k(k)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    annualization_periods = (
        TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * PERIODS_PER_HOUR
    )
    annualization_factor = float(np.sqrt(annualization_periods))

    rows: list[dict[str, Any]] = []
    series_report: dict[str, Any] = {}
    for item in paths.inputs():
        item_rows, item_report = _compute_series_metrics(
            item,
            k=k,
            annualization_factor=annualization_factor,
        )
        rows.extend(item_rows)
        series_report[item.name] = item_report

    output = pd.DataFrame(rows)
    write_csv(output, paths.output_csv, index=False)

    report = {
        "K": k,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "annualization_assumptions": {
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "trading_hours_per_day": TRADING_HOURS_PER_DAY,
            "periods_per_hour": PERIODS_PER_HOUR,
            "periods_per_year": annualization_periods,
            "annualization_factor": annualization_factor,
        },
        "output_csv": str(paths.output_csv),
        "series": series_report,
    }
    write_json(paths.report_json, report)
    return report


def _compute_series_metrics(
    item: VolatilityInput,
    k: int,
    annualization_factor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    components = decomposition_components(k, include_original=False)
    columns = [ORIGINAL, *components]
    frame = pd.read_csv(item.decomposition_csv, usecols=columns)
    if frame.empty:
        raise ValueError(f"Decomposition file is empty: {item.decomposition_csv}")

    n = len(frame)
    original = frame[ORIGINAL].astype(float).to_numpy()
    require_finite_array(original, f"Original series in {item.decomposition_csv}")

    component_metrics = []
    detail_energies: dict[str, float] = {}
    component_energies: dict[str, float] = {}
    component_means: dict[str, float] = {}

    for component in components:
        values = frame[component].astype(float).to_numpy()
        require_finite_array(values, f"Component {component} in {item.decomposition_csv}")

        energy = float(np.dot(values, values))
        mean = float(np.mean(values))
        rms = float(np.sqrt(energy / n))
        kind = component_type(component)
        scale = component_scale(component)
        scale_minutes = component_scale_minutes(component, BASE_INTERVAL_MINUTES)
        metrics = {
            SERIES: item.name,
            "component": component,
            "k": scale,
            COMPONENT_TYPE: kind,
            "scale_minutes": scale_minutes,
            "scale_days": scale_minutes / (60 * 24),
            "energy": energy,
            "rms_volatility": rms,
            "annualized_rms_volatility": rms * annualization_factor,
            "detail_energy_share": np.nan,
            "total_component_energy_share": np.nan,
        }
        component_metrics.append(metrics)
        component_energies[component] = energy
        component_means[component] = mean
        if kind == "detail":
            detail_energies[component] = energy

    detail_energy_sum = float(sum(detail_energies.values()))
    total_component_energy = float(sum(component_energies.values()))
    if detail_energy_sum <= 0:
        raise ValueError(f"Detail energy sum is non-positive for {item.name}")
    if total_component_energy <= 0:
        raise ValueError(f"Total component energy is non-positive for {item.name}")

    for metrics in component_metrics:
        component = metrics["component"]
        if metrics[COMPONENT_TYPE] == "detail":
            metrics["detail_energy_share"] = (
                component_energies[component] / detail_energy_sum
            )
        metrics["total_component_energy_share"] = (
            component_energies[component] / total_component_energy
        )

    original_energy = float(np.dot(original, original))
    approximation_component = f"A_{k:02d}"
    report = {
        "input_csv": str(item.decomposition_csv),
        "N": int(n),
        "original_energy": original_energy,
        "detail_energy_sum": detail_energy_sum,
        "approximation_energy": component_energies[approximation_component],
        "total_component_energy": total_component_energy,
        "energy_reconstruction_gap": original_energy - total_component_energy,
        "detail_energy_share_sum": float(
            sum(
                row["detail_energy_share"]
                for row in component_metrics
                if row[COMPONENT_TYPE] == "detail"
            )
        ),
        "total_component_energy_share_sum": float(
            sum(row["total_component_energy_share"] for row in component_metrics)
        ),
        "component_means": component_means,
    }
    return component_metrics, report
