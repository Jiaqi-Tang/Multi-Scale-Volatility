"""High-level entropy plot orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multi_scale_volatility.config.constants import DEFAULT_K
from multi_scale_volatility.config.metric_columns import NORMALIZED_ENTROPY, PERMUTATION_ENTROPY
from multi_scale_volatility.config.paths import (
    ENTROPY_GAPS_CSV,
    ENTROPY_PLOTS_DIR,
    ENTROPY_REPORT_JSON,
    LAYER_ENTROPY_CSV,
)
from multi_scale_volatility.config.series import SERIES_FINAL, SERIES_GAUSSIAN, SERIES_ORDER, SERIES_SHUFFLE
from multi_scale_volatility.plotting.primitives import (
    plot_entropy_gaps,
    plot_entropy_metric,
    plot_entropy_pattern_distribution_grid,
)
from multi_scale_volatility.plotting.readers import (
    read_entropy_gaps,
    read_entropy_pattern_counts,
    read_layer_entropy,
)
from multi_scale_volatility.utils.validation import require_positive_k


@dataclass(frozen=True)
class EntropyPlotPaths:
    layer_entropy_csv: Path = LAYER_ENTROPY_CSV
    entropy_gaps_csv: Path = ENTROPY_GAPS_CSV
    entropy_report_json: Path = ENTROPY_REPORT_JSON
    output_dir: Path = ENTROPY_PLOTS_DIR


def create_entropy_plots(
    paths: EntropyPlotPaths | None = None,
    k: int = DEFAULT_K,
) -> list[Path]:
    paths = paths or EntropyPlotPaths()
    require_positive_k(k)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    layer_entropy = read_layer_entropy(paths.layer_entropy_csv, k)
    entropy_gaps = read_entropy_gaps(paths.entropy_gaps_csv, k)
    pattern_counts = read_entropy_pattern_counts(paths.entropy_report_json, k)

    outputs = [
        plot_entropy_metric(
            layer_entropy,
            paths.output_dir / "permutation_entropy.png",
            metric=PERMUTATION_ENTROPY,
            title="Permutation Entropy by Decomposition Component",
            ylabel="Permutation entropy",
            k=k,
        ),
        plot_entropy_metric(
            layer_entropy,
            paths.output_dir / "normalized_entropy.png",
            metric=NORMALIZED_ENTROPY,
            title="Normalized Permutation Entropy by Decomposition Component",
            ylabel="Normalized entropy",
            k=k,
        ),
        plot_entropy_gaps(
            entropy_gaps,
            paths.output_dir / "entropy_gaps.png",
            k=k,
        ),
    ]
    pattern_filenames = {
        SERIES_FINAL: "final_pattern_distribution.png",
        SERIES_SHUFFLE: "shuffle_pattern_distribution.png",
        SERIES_GAUSSIAN: "gaussian_pattern_distribution.png",
    }
    for series in SERIES_ORDER:
        outputs.append(
            plot_entropy_pattern_distribution_grid(
                pattern_counts[series],
                paths.output_dir / pattern_filenames[series],
                series=series,
                k=k,
            )
        )
    return outputs
