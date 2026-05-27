"""Permutation entropy metrics for decomposition components."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multi_scale_volatility.components import component_specs
from multi_scale_volatility.scale_utils import compress_component
from multi_scale_volatility.config.bundles import SeriesBundle
from multi_scale_volatility.config.columns import COMPONENT, COMPONENT_TYPE, SERIES
from multi_scale_volatility.config.constants import BASE_INTERVAL_MINUTES, DEFAULT_K
from multi_scale_volatility.config.metric_columns import (
    EFFECTIVE_N,
    ENTROPY_GAP_GAUSSIAN,
    ENTROPY_GAP_SHUFFLE,
    FINAL_NORMALIZED_ENTROPY,
    GAUSSIAN_NORMALIZED_ENTROPY,
    K,
    NORMALIZED_ENTROPY,
    ORDINAL_WINDOWS,
    PERMUTATION_ENTROPY,
    REPEAT_LENGTH,
    SCALE_DAYS,
    SCALE_MINUTES,
    SHUFFLE_NORMALIZED_ENTROPY,
)
from multi_scale_volatility.config.paths import (
    ENTROPY_GAPS_CSV,
    ENTROPY_REPORT_JSON,
    ENTROPY_RESULTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    GAUSSIAN_DECOMPOSITION_CSV,
    LAYER_ENTROPY_CSV,
    SHUFFLE_DECOMPOSITION_CSV,
)
from multi_scale_volatility.config.path_utils import resolve_artifact_path
from multi_scale_volatility.config.series import (
    SERIES_FINAL,
    SERIES_GAUSSIAN,
    SERIES_ORDER,
    SERIES_SHUFFLE,
)
from multi_scale_volatility.config.schemas import ENTROPY_GAP_INDEX_COLUMNS
from multi_scale_volatility.utils.artifact_io import write_csv
from multi_scale_volatility.utils.json_utils import write_json
from multi_scale_volatility.utils.validation import require_finite_array, require_positive_k

EMBEDDING_DIMENSION = 3
DELAY = 1
JITTER_SEED = 314
JITTER_MAGNITUDE = 1e-10


@dataclass(frozen=True)
class EntropyInput:
    name: str
    decomposition_csv: Path


@dataclass(frozen=True)
class EntropyPaths:
    final_decomposition_csv: Path = FINAL_DECOMPOSITION_CSV
    shuffle_decomposition_csv: Path = SHUFFLE_DECOMPOSITION_CSV
    gaussian_decomposition_csv: Path = GAUSSIAN_DECOMPOSITION_CSV
    output_dir: Path = ENTROPY_RESULTS_DIR

    @property
    def layer_entropy_csv(self) -> Path:
        return resolve_artifact_path(
            self.output_dir,
            ENTROPY_RESULTS_DIR,
            LAYER_ENTROPY_CSV,
        )

    @property
    def entropy_gaps_csv(self) -> Path:
        return resolve_artifact_path(
            self.output_dir,
            ENTROPY_RESULTS_DIR,
            ENTROPY_GAPS_CSV,
        )

    @property
    def report_json(self) -> Path:
        return resolve_artifact_path(
            self.output_dir,
            ENTROPY_RESULTS_DIR,
            ENTROPY_REPORT_JSON,
        )

    def input_bundle(self) -> SeriesBundle[Path]:
        return SeriesBundle(
            final=self.final_decomposition_csv,
            shuffle=self.shuffle_decomposition_csv,
            gaussian=self.gaussian_decomposition_csv,
        )

    def inputs(self) -> list[EntropyInput]:
        return [
            EntropyInput(name, decomposition_csv)
            for name, decomposition_csv in self.input_bundle()
        ]


def compute_entropy_metrics(
    paths: EntropyPaths | None = None,
    k: int = DEFAULT_K,
    embedding_dimension: int = EMBEDDING_DIMENSION,
    delay: int = DELAY,
    jitter_seed: int = JITTER_SEED,
    jitter_magnitude: float = JITTER_MAGNITUDE,
) -> dict[str, Any]:
    paths = paths or EntropyPaths()
    require_positive_k(k)
    if embedding_dimension < 2:
        raise ValueError("embedding_dimension must be at least 2")
    if delay < 1:
        raise ValueError("delay must be at least 1")
    if jitter_magnitude < 0:
        raise ValueError("jitter_magnitude must be non-negative")

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pattern_counts: dict[str, dict[str, dict[str, int]]] = {}
    series_report: dict[str, Any] = {}
    for item in paths.inputs():
        item_rows, item_counts, item_report = _compute_series_entropy(
            item,
            k=k,
            embedding_dimension=embedding_dimension,
            delay=delay,
            jitter_seed=jitter_seed,
            jitter_magnitude=jitter_magnitude,
        )
        rows.extend(item_rows)
        pattern_counts[item.name] = item_counts
        series_report[item.name] = item_report

    layer_entropy = pd.DataFrame(rows)
    write_csv(layer_entropy, paths.layer_entropy_csv, index=False)

    entropy_gaps = _compute_entropy_gaps(layer_entropy)
    write_csv(entropy_gaps, paths.entropy_gaps_csv, index=False)

    report = {
        "K": k,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "embedding_dimension": embedding_dimension,
        "delay": delay,
        "log_base": "natural",
        "normalization_denominator": math.log(math.factorial(embedding_dimension)),
        "jitter_seed": jitter_seed,
        "jitter_magnitude": jitter_magnitude,
        "compression_rule": (
            "Deterministic repeated block values are removed before entropy "
            "calculation using component-specific repeat lengths."
        ),
        "layer_entropy_csv": str(paths.layer_entropy_csv),
        "entropy_gaps_csv": str(paths.entropy_gaps_csv),
        "series": series_report,
        "pattern_counts": pattern_counts,
    }
    write_json(paths.report_json, report)
    return report


def _compute_series_entropy(
    item: EntropyInput,
    k: int,
    embedding_dimension: int,
    delay: int,
    jitter_seed: int,
    jitter_magnitude: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    specs = component_specs(k, include_original=False,
                            base_interval_minutes=BASE_INTERVAL_MINUTES)
    frame = pd.read_csv(item.decomposition_csv, usecols=[
                        spec.name for spec in specs])
    if frame.empty:
        raise ValueError(
            f"Decomposition file is empty: {item.decomposition_csv}")

    rows: list[dict[str, Any]] = []
    pattern_counts: dict[str, dict[str, int]] = {}
    component_report: dict[str, Any] = {}

    for spec in specs:
        component = spec.name
        values = frame[component].astype(float).to_numpy()
        require_finite_array(
            values, f"Component {component} in {item.decomposition_csv}")

        compressed = compress_component(values, component)
        if len(compressed) < embedding_dimension:
            raise ValueError(
                f"Compressed component too short for entropy: {item.name} {component}"
            )

        component_seed = _component_jitter_seed(
            jitter_seed, item.name, component)
        jittered = _add_jitter(compressed, component_seed, jitter_magnitude)
        entropy_result = _permutation_entropy(
            jittered, embedding_dimension, delay)

        rows.append(
            {
                SERIES: item.name,
                COMPONENT: component,
                K: spec.scale,
                COMPONENT_TYPE: spec.kind,
                SCALE_MINUTES: spec.scale_minutes,
                SCALE_DAYS: spec.scale_days,
                REPEAT_LENGTH: spec.repeat_length,
                EFFECTIVE_N: len(compressed),
                ORDINAL_WINDOWS: entropy_result[ORDINAL_WINDOWS],
                PERMUTATION_ENTROPY: entropy_result[PERMUTATION_ENTROPY],
                NORMALIZED_ENTROPY: entropy_result[NORMALIZED_ENTROPY],
            }
        )
        pattern_counts[component] = entropy_result["pattern_counts"]
        component_report[component] = {
            REPEAT_LENGTH: spec.repeat_length,
            "expanded_n": int(len(values)),
            EFFECTIVE_N: int(len(compressed)),
            ORDINAL_WINDOWS: entropy_result[ORDINAL_WINDOWS],
            "component_jitter_seed": component_seed,
            "compressed_unique_values": int(pd.Series(compressed).nunique()),
        }

    report = {
        "input_csv": str(item.decomposition_csv),
        "expanded_N": int(len(frame)),
        "components": component_report,
    }
    return rows, pattern_counts, report


def _permutation_entropy(
    values: np.ndarray,
    embedding_dimension: int,
    delay: int,
) -> dict[str, Any]:
    ordinal_windows = len(values) - (embedding_dimension - 1) * delay
    if ordinal_windows <= 0:
        raise ValueError("Series is too short for permutation entropy")

    permutations = list(itertools.permutations(range(embedding_dimension)))
    counts = {"".join(str(index) for index in pattern)              : 0 for pattern in permutations}

    for start in range(ordinal_windows):
        window = values[start: start + embedding_dimension * delay: delay]
        pattern = tuple(np.argsort(window, kind="mergesort"))
        counts["".join(str(index) for index in pattern)] += 1

    probabilities = np.array(
        [count / ordinal_windows for count in counts.values() if count > 0],
        dtype=float,
    )
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = float(
        entropy / math.log(math.factorial(embedding_dimension)))
    return {
        ORDINAL_WINDOWS: int(ordinal_windows),
        "pattern_counts": counts,
        PERMUTATION_ENTROPY: entropy,
        NORMALIZED_ENTROPY: normalized_entropy,
    }


def _compute_entropy_gaps(layer_entropy: pd.DataFrame) -> pd.DataFrame:
    index_columns = list(ENTROPY_GAP_INDEX_COLUMNS)
    wide = layer_entropy.pivot_table(
        index=index_columns,
        columns=SERIES,
        values=NORMALIZED_ENTROPY,
        aggfunc="first",
    ).reset_index()
    required_series = list(SERIES_ORDER)
    missing_series = [
        series for series in required_series if series not in wide.columns]
    if missing_series:
        raise ValueError(
            f"Missing entropy series for gap calculation: {missing_series}")

    wide = wide.rename(
        columns={
            SERIES_FINAL: FINAL_NORMALIZED_ENTROPY,
            SERIES_SHUFFLE: SHUFFLE_NORMALIZED_ENTROPY,
            SERIES_GAUSSIAN: GAUSSIAN_NORMALIZED_ENTROPY,
        }
    )
    wide[ENTROPY_GAP_SHUFFLE] = (
        wide[SHUFFLE_NORMALIZED_ENTROPY] - wide[FINAL_NORMALIZED_ENTROPY]
    )
    wide[ENTROPY_GAP_GAUSSIAN] = (
        wide[GAUSSIAN_NORMALIZED_ENTROPY] - wide[FINAL_NORMALIZED_ENTROPY]
    )
    return wide[
        [
            *index_columns,
            FINAL_NORMALIZED_ENTROPY,
            SHUFFLE_NORMALIZED_ENTROPY,
            GAUSSIAN_NORMALIZED_ENTROPY,
            ENTROPY_GAP_SHUFFLE,
            ENTROPY_GAP_GAUSSIAN,
        ]
    ].sort_values(K)


def _add_jitter(
    values: np.ndarray,
    seed: int,
    magnitude: float,
) -> np.ndarray:
    if magnitude == 0:
        return values.copy()
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-magnitude, magnitude, size=len(values))
    return values + jitter


def _component_jitter_seed(base_seed: int, series: str, component: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{series}:{component}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)
