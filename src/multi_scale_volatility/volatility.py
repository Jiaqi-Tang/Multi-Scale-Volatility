"""Volatility and energy metrics for decomposition components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.components import component_specs
from multi_scale_volatility.config.bundles import SeriesBundle
from multi_scale_volatility.config.columns import COMPONENT, COMPONENT_TYPE, ORIGINAL, SERIES
from multi_scale_volatility.config.constants import (
    BASE_INTERVAL_MINUTES,
    DEFAULT_K,
    PERIODS_PER_HOUR,
    TRADING_DAYS_PER_YEAR,
    TRADING_HOURS_PER_DAY,
)
from multi_scale_volatility.config.paths import (
    FINAL_DECOMPOSITION_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    SHUFFLE_DECOMPOSITION_CSV,
    VOLATILITY_CSV,
    VOLATILITY_REPORT_JSON,
    VOLATILITY_RESULTS_DIR,
)
from multi_scale_volatility.config.path_utils import resolve_artifact_path
from multi_scale_volatility.config.metric_columns import (
    ANNUALIZED_RMS_VOLATILITY,
    APPROXIMATION_ENERGY,
    DETAIL_ENERGY_SHARE,
    DETAIL_ENERGY_SHARE_SUM,
    DETAIL_ENERGY_SUM,
    ENERGY,
    ENERGY_RECONSTRUCTION_GAP,
    K,
    ORIGINAL_ENERGY,
    RMS_VOLATILITY,
    SCALE_DAYS,
    SCALE_MINUTES,
    TOTAL_COMPONENT_ENERGY,
    TOTAL_COMPONENT_ENERGY_SHARE,
    TOTAL_COMPONENT_ENERGY_SHARE_SUM,
)
from multi_scale_volatility.utils.artifact_io import write_csv
from multi_scale_volatility.utils.json_utils import write_json
from multi_scale_volatility.utils.validation import require_finite_array, require_positive_k


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
        return resolve_artifact_path(
            self.output_dir,
            VOLATILITY_RESULTS_DIR,
            VOLATILITY_CSV,
        )

    @property
    def report_json(self) -> Path:
        return resolve_artifact_path(
            self.output_dir,
            VOLATILITY_RESULTS_DIR,
            VOLATILITY_REPORT_JSON,
        )

    def input_bundle(self) -> SeriesBundle[Path]:
        return SeriesBundle(
            final=self.final_decomposition_csv,
            shuffle=self.shuffle_decomposition_csv,
            gaussian=self.gaussian_decomposition_csv,
        )

    def inputs(self) -> list[VolatilityInput]:
        return [
            VolatilityInput(name, decomposition_csv)
            for name, decomposition_csv in self.input_bundle()
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
    specs = component_specs(k, include_original=False,
                            base_interval_minutes=BASE_INTERVAL_MINUTES)
    columns = [ORIGINAL, *(spec.name for spec in specs)]
    frame = pd.read_csv(item.decomposition_csv, usecols=columns)
    if frame.empty:
        raise ValueError(
            f"Decomposition file is empty: {item.decomposition_csv}")

    n = len(frame)
    original = frame[ORIGINAL].astype(float).to_numpy()
    require_finite_array(
        original, f"Original series in {item.decomposition_csv}")

    component_metrics = []
    detail_energies: dict[str, float] = {}
    component_energies: dict[str, float] = {}
    component_means: dict[str, float] = {}

    for spec in specs:
        component = spec.name
        values = frame[component].astype(float).to_numpy()
        require_finite_array(
            values, f"Component {component} in {item.decomposition_csv}")

        energy = float(np.dot(values, values))
        mean = float(np.mean(values))
        rms = float(np.sqrt(energy / n))
        metrics = {
            SERIES: item.name,
            COMPONENT: component,
            K: spec.scale,
            COMPONENT_TYPE: spec.kind,
            SCALE_MINUTES: spec.scale_minutes,
            SCALE_DAYS: spec.scale_days,
            ENERGY: energy,
            RMS_VOLATILITY: rms,
            ANNUALIZED_RMS_VOLATILITY: rms * annualization_factor,
            DETAIL_ENERGY_SHARE: np.nan,
            TOTAL_COMPONENT_ENERGY_SHARE: np.nan,
        }
        component_metrics.append(metrics)
        component_energies[component] = energy
        component_means[component] = mean
        if spec.kind == "detail":
            detail_energies[component] = energy

    detail_energy_sum = float(sum(detail_energies.values()))
    total_component_energy = float(sum(component_energies.values()))
    if detail_energy_sum <= 0:
        raise ValueError(f"Detail energy sum is non-positive for {item.name}")
    if total_component_energy <= 0:
        raise ValueError(
            f"Total component energy is non-positive for {item.name}")

    for metrics in component_metrics:
        component = metrics[COMPONENT]
        if metrics[COMPONENT_TYPE] == "detail":
            metrics[DETAIL_ENERGY_SHARE] = (
                component_energies[component] / detail_energy_sum
            )
        metrics[TOTAL_COMPONENT_ENERGY_SHARE] = (
            component_energies[component] / total_component_energy
        )

    original_energy = float(np.dot(original, original))
    approximation_component = f"A_{k:02d}"
    report = {
        "input_csv": str(item.decomposition_csv),
        "N": int(n),
        ORIGINAL_ENERGY: original_energy,
        DETAIL_ENERGY_SUM: detail_energy_sum,
        APPROXIMATION_ENERGY: component_energies[approximation_component],
        TOTAL_COMPONENT_ENERGY: total_component_energy,
        ENERGY_RECONSTRUCTION_GAP: original_energy - total_component_energy,
        DETAIL_ENERGY_SHARE_SUM: float(
            sum(
                row[DETAIL_ENERGY_SHARE]
                for row in component_metrics
                if row[COMPONENT_TYPE] == "detail"
            )
        ),
        TOTAL_COMPONENT_ENERGY_SHARE_SUM: float(
            sum(row[TOTAL_COMPONENT_ENERGY_SHARE] for row in component_metrics)
        ),
        "component_means": component_means,
    }
    return component_metrics, report
