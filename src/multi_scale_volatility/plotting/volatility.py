"""High-level volatility plot orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multi_scale_volatility.config.columns import COMPONENT_TYPE
from multi_scale_volatility.config.constants import DEFAULT_K
from multi_scale_volatility.config.metric_columns import (
    ANNUALIZED_RMS_VOLATILITY,
    DETAIL_ENERGY_SHARE,
    RMS_VOLATILITY,
    TOTAL_COMPONENT_ENERGY_SHARE,
)
from multi_scale_volatility.config.paths import VOLATILITY_CSV, VOLATILITY_PLOTS_DIR
from multi_scale_volatility.plotting.primitives import (
    plot_volatility_difference_metric,
    plot_volatility_metric,
)
from multi_scale_volatility.plotting.readers import read_volatility
from multi_scale_volatility.utils.validation import require_positive_k


@dataclass(frozen=True)
class VolatilityPlotPaths:
    volatility_csv: Path = VOLATILITY_CSV
    output_dir: Path = VOLATILITY_PLOTS_DIR


def create_volatility_plots(
    paths: VolatilityPlotPaths | None = None,
    k: int = DEFAULT_K,
) -> list[Path]:
    paths = paths or VolatilityPlotPaths()
    require_positive_k(k)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_volatility(paths.volatility_csv, k)

    return [
        plot_volatility_metric(
            frame[frame[COMPONENT_TYPE] == "detail"],
            paths.output_dir / "detail_energy_share.png",
            metric=DETAIL_ENERGY_SHARE,
            title="Detail Energy Share by Decomposition Component",
            ylabel="Detail energy share",
            k=k,
            include_approximation=False,
        ),
        plot_volatility_metric(
            frame,
            paths.output_dir / "total_component_energy_share.png",
            metric=TOTAL_COMPONENT_ENERGY_SHARE,
            title="Total Component Energy Share by Decomposition Component",
            ylabel="Total component energy share",
            k=k,
            include_approximation=True,
        ),
        plot_volatility_metric(
            frame,
            paths.output_dir / "rms_volatility.png",
            metric=RMS_VOLATILITY,
            title="RMS Volatility by Decomposition Component",
            ylabel="RMS volatility",
            k=k,
            include_approximation=True,
        ),
        plot_volatility_metric(
            frame,
            paths.output_dir / "annualized_rms_volatility.png",
            metric=ANNUALIZED_RMS_VOLATILITY,
            title="Annualized RMS Volatility by Decomposition Component",
            ylabel="Annualized RMS volatility",
            k=k,
            include_approximation=True,
        ),
        plot_volatility_difference_metric(
            frame[frame[COMPONENT_TYPE] == "detail"],
            paths.output_dir / "detail_energy_share_difference.png",
            metric=DETAIL_ENERGY_SHARE,
            title="Detail Energy Share Difference from Baselines",
            ylabel="Final minus baseline",
            k=k,
            include_approximation=False,
        ),
        plot_volatility_difference_metric(
            frame,
            paths.output_dir / "total_component_energy_share_difference.png",
            metric=TOTAL_COMPONENT_ENERGY_SHARE,
            title="Total Component Energy Share Difference from Baselines",
            ylabel="Final minus baseline",
            k=k,
            include_approximation=True,
        ),
        plot_volatility_difference_metric(
            frame,
            paths.output_dir / "rms_volatility_difference.png",
            metric=RMS_VOLATILITY,
            title="RMS Volatility Difference from Baselines",
            ylabel="Final minus baseline",
            k=k,
            include_approximation=True,
        ),
    ]
