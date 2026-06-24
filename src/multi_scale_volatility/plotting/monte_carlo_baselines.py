"""V1.1 plots using Monte Carlo baseline envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_scale_volatility.config.names import COMPONENT, COMPONENT_TYPE, LOG_RETURN
from multi_scale_volatility.config.names import DEFAULT_K
from multi_scale_volatility.config.names import (
    ANNUALIZED_RMS_VOLATILITY,
    DETAIL_ENERGY_SHARE,
    NORMALIZED_ENTROPY,
    PERMUTATION_ENTROPY,
    RMS_VOLATILITY,
    TOTAL_COMPONENT_ENERGY_SHARE,
)
from multi_scale_volatility.config.paths import (
    DECOMPOSITION_PLOTS_DIR,
    EDA_PLOTS_DIR,
    ENTROPY_PLOTS_DIR,
    FINAL_DECOMPOSITION_CSV,
    FINAL_RETURNS_CSV,
    MC_ABS_COMPONENT_CORRELATION_EMPIRICAL_COMPARISON_CSV,
    MC_ABS_COMPONENT_CORRELATION_SUMMARY_CSV,
    MC_ACF_EMPIRICAL_COMPARISON_CSV,
    MC_ACF_SUMMARY_CSV,
    MC_COMPONENT_ACF_EMPIRICAL_COMPARISON_CSV,
    MC_COMPONENT_ACF_SUMMARY_CSV,
    MC_LAYER_ENTROPY_EMPIRICAL_COMPARISON_CSV,
    MC_LAYER_ENTROPY_SUMMARY_CSV,
    MC_LAYER_VOLATILITY_EMPIRICAL_COMPARISON_CSV,
    MC_LAYER_VOLATILITY_SUMMARY_CSV,
    MEMO_PLOTS_DIR,
    MONTE_CARLO_BASELINE_AUDIT_CSV,
    MONTE_CARLO_BASELINES_RESULTS_DIR,
    VOLATILITY_PLOTS_DIR,
)
from multi_scale_volatility.plotting.memo import (
    plot_memo_decomposition_example,
    plot_memo_return_distribution,
)
from multi_scale_volatility.plotting.save import save_figure
from multi_scale_volatility.plotting.style import (
    FIGURE_DPI,
    FINAL_COLOR,
    GAUSSIAN_COLOR,
    SHUFFLE_COLOR,
)
from multi_scale_volatility.components import decomposition_components
from multi_scale_volatility.stats import absolute_component_correlation, autocorrelation
from multi_scale_volatility.utils.validation import require_positive_k


@dataclass(frozen=True)
class MonteCarloBaselinePlotPaths:
    results_dir: Path = MONTE_CARLO_BASELINES_RESULTS_DIR
    audit_csv: Path = MONTE_CARLO_BASELINE_AUDIT_CSV
    final_returns_csv: Path = FINAL_RETURNS_CSV
    final_decomposition_csv: Path = FINAL_DECOMPOSITION_CSV
    decomposition_output_dir: Path = DECOMPOSITION_PLOTS_DIR
    volatility_output_dir: Path = VOLATILITY_PLOTS_DIR
    entropy_output_dir: Path = ENTROPY_PLOTS_DIR
    eda_output_dir: Path = EDA_PLOTS_DIR
    memo_output_dir: Path = MEMO_PLOTS_DIR
    correlation_output_dir: Path = Path("plots/results/correlation")

    @property
    def volatility_summary_csv(self) -> Path:
        return self.results_dir / MC_LAYER_VOLATILITY_SUMMARY_CSV.name

    @property
    def volatility_comparison_csv(self) -> Path:
        return self.results_dir / MC_LAYER_VOLATILITY_EMPIRICAL_COMPARISON_CSV.name

    @property
    def entropy_summary_csv(self) -> Path:
        return self.results_dir / MC_LAYER_ENTROPY_SUMMARY_CSV.name

    @property
    def entropy_comparison_csv(self) -> Path:
        return self.results_dir / MC_LAYER_ENTROPY_EMPIRICAL_COMPARISON_CSV.name

    @property
    def acf_summary_csv(self) -> Path:
        return self.results_dir / MC_ACF_SUMMARY_CSV.name

    @property
    def acf_comparison_csv(self) -> Path:
        return self.results_dir / MC_ACF_EMPIRICAL_COMPARISON_CSV.name

    @property
    def component_acf_summary_csv(self) -> Path:
        return self.results_dir / MC_COMPONENT_ACF_SUMMARY_CSV.name

    @property
    def component_acf_comparison_csv(self) -> Path:
        return self.results_dir / MC_COMPONENT_ACF_EMPIRICAL_COMPARISON_CSV.name

    @property
    def corr_summary_csv(self) -> Path:
        return self.results_dir / MC_ABS_COMPONENT_CORRELATION_SUMMARY_CSV.name

    @property
    def corr_comparison_csv(self) -> Path:
        return self.results_dir / MC_ABS_COMPONENT_CORRELATION_EMPIRICAL_COMPARISON_CSV.name


def create_v11_memo_plots(
    paths: MonteCarloBaselinePlotPaths | None = None,
    k: int = DEFAULT_K,
) -> list[Path]:
    paths = paths or MonteCarloBaselinePlotPaths()
    require_positive_k(k)
    paths.memo_output_dir.mkdir(parents=True, exist_ok=True)

    final_returns = pd.read_csv(paths.final_returns_csv, usecols=[LOG_RETURN])[
        LOG_RETURN
    ].astype(float).to_numpy()
    final_decomposition = pd.read_csv(paths.final_decomposition_csv)
    audit = pd.read_csv(paths.audit_csv)
    gaussian_record = (
        audit[audit["baseline_type"] == "gaussian"]
        .sort_values("simulation_id")
        .iloc[0]
    )
    gaussian_returns = pd.read_parquet(
        gaussian_record["return_parquet"],
        columns=[LOG_RETURN],
    )[LOG_RETURN].astype(float).to_numpy()

    outputs = [
        plot_memo_decomposition_example(
            final_decomposition,
            paths.memo_output_dir / "figure_01_decomposition_example.png",
        ),
        plot_memo_return_distribution(
            final_returns,
            gaussian_returns,
            paths.memo_output_dir / "figure_02_return_distribution.png",
        ),
    ]
    outputs.extend(create_monte_carlo_baseline_plots(paths, k=k))
    return outputs


def create_monte_carlo_baseline_plots(
    paths: MonteCarloBaselinePlotPaths | None = None,
    k: int = DEFAULT_K,
) -> list[Path]:
    paths = paths or MonteCarloBaselinePlotPaths()
    require_positive_k(k)
    for directory in [
        paths.volatility_output_dir,
        paths.entropy_output_dir,
        paths.eda_output_dir,
        paths.decomposition_output_dir,
        paths.memo_output_dir,
        paths.correlation_output_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    volatility_summary = pd.read_csv(paths.volatility_summary_csv)
    volatility_comparison = pd.read_csv(paths.volatility_comparison_csv)
    entropy_summary = pd.read_csv(paths.entropy_summary_csv)
    entropy_comparison = pd.read_csv(paths.entropy_comparison_csv)
    acf_summary = pd.read_csv(paths.acf_summary_csv)
    acf_comparison = pd.read_csv(paths.acf_comparison_csv)
    corr_comparison = pd.read_csv(paths.corr_comparison_csv)

    outputs: list[Path] = []
    details = [f"D_{scale:02d}" for scale in range(1, k + 1)]
    all_components = [*details, f"A_{k:02d}"]

    outputs.extend(
        [
            plot_metric_envelope(
                volatility_summary,
                volatility_comparison,
                paths.volatility_output_dir / "detail_energy_share.png",
                metric=DETAIL_ENERGY_SHARE,
                components=details,
                title="Detail Energy Share with Monte Carlo Baseline Envelopes",
                ylabel="Detail energy share",
            ),
            plot_metric_excess(
                volatility_comparison,
                paths.volatility_output_dir / "detail_energy_share_difference.png",
                metric=DETAIL_ENERGY_SHARE,
                components=details,
                title="Detail Energy Share Excess vs Baseline Medians",
                ylabel="EUR/USD minus baseline median",
            ),
            plot_metric_envelope(
                volatility_summary,
                volatility_comparison,
                paths.volatility_output_dir / "total_component_energy_share.png",
                metric=TOTAL_COMPONENT_ENERGY_SHARE,
                components=all_components,
                title="Total Component Energy Share with Monte Carlo Baseline Envelopes",
                ylabel="Total component energy share",
            ),
            plot_metric_excess(
                volatility_comparison,
                paths.volatility_output_dir / "total_component_energy_share_difference.png",
                metric=TOTAL_COMPONENT_ENERGY_SHARE,
                components=all_components,
                title="Total Component Energy Share Excess vs Baseline Medians",
                ylabel="EUR/USD minus baseline median",
            ),
            plot_metric_envelope(
                volatility_summary,
                volatility_comparison,
                paths.volatility_output_dir / "rms_volatility.png",
                metric=RMS_VOLATILITY,
                components=all_components,
                title="RMS Volatility with Monte Carlo Baseline Envelopes",
                ylabel="RMS volatility",
            ),
            plot_metric_excess(
                volatility_comparison,
                paths.volatility_output_dir / "rms_volatility_difference.png",
                metric=RMS_VOLATILITY,
                components=all_components,
                title="RMS Volatility Excess vs Baseline Medians",
                ylabel="EUR/USD minus baseline median",
            ),
            plot_metric_envelope(
                volatility_summary,
                volatility_comparison,
                paths.volatility_output_dir / "annualized_rms_volatility.png",
                metric=ANNUALIZED_RMS_VOLATILITY,
                components=all_components,
                title="Annualized RMS Volatility with Monte Carlo Baseline Envelopes",
                ylabel="Annualized RMS volatility",
            ),
        ]
    )

    outputs.extend(
        [
            plot_metric_envelope(
                entropy_summary,
                entropy_comparison,
                paths.entropy_output_dir / "permutation_entropy.png",
                metric=PERMUTATION_ENTROPY,
                components=all_components,
                title="Permutation Entropy with Monte Carlo Baseline Envelopes",
                ylabel="Permutation entropy",
            ),
            plot_metric_envelope(
                entropy_summary,
                entropy_comparison,
                paths.entropy_output_dir / "normalized_entropy.png",
                metric=NORMALIZED_ENTROPY,
                components=all_components,
                title="Normalized Entropy with Monte Carlo Baseline Envelopes",
                ylabel="Normalized entropy",
            ),
            plot_metric_excess(
                entropy_comparison,
                paths.entropy_output_dir / "entropy_gaps.png",
                metric=NORMALIZED_ENTROPY,
                components=all_components,
                title="Normalized Entropy Excess vs Baseline Medians",
                ylabel="EUR/USD minus baseline median",
            ),
        ]
    )

    outputs.extend(
        [
            plot_acf_envelope(
                acf_summary,
                acf_comparison,
                paths.eda_output_dir / "final_vs_baselines_returns_acf.png",
                acf_kind="return",
                max_lag=288,
                title="Return ACF with Monte Carlo Baseline Envelopes",
            ),
            plot_acf_envelope(
                acf_summary,
                acf_comparison,
                paths.eda_output_dir / "final_vs_baselines_abs_returns_acf.png",
                acf_kind="absolute_return",
                max_lag=288,
                title="Absolute Return ACF with Monte Carlo Baseline Envelopes",
            ),
            plot_acf_envelope(
                acf_summary,
                acf_comparison,
                paths.memo_output_dir / "figure_03_abs_return_acf.png",
                acf_kind="absolute_return",
                max_lag=1440,
                title="Autocorrelation of Absolute 5m Returns",
            ),
            plot_memo_energy_profile(
                volatility_summary,
                volatility_comparison,
                paths.memo_output_dir / "figure_04_energy_profile.png",
                components=details,
            ),
            plot_correlation_memo(
                paths.final_decomposition_csv,
                corr_comparison,
                paths.memo_output_dir / "figure_05_cross_scale_correlation.png",
                k=k,
            ),
            plot_memo_entropy_profile(
                entropy_summary,
                entropy_comparison,
                paths.memo_output_dir / "figure_06_entropy_profile.png",
                components=all_components,
            ),
        ]
    )

    outputs.extend(
        plot_correlation_result_plots(
            paths.final_decomposition_csv,
            corr_comparison,
            paths.correlation_output_dir,
            k=k,
        )
    )
    if paths.component_acf_summary_csv.exists() and paths.component_acf_comparison_csv.exists():
        component_acf_summary = pd.read_csv(paths.component_acf_summary_csv)
        component_acf_comparison = pd.read_csv(paths.component_acf_comparison_csv)
        outputs.extend(
            create_component_acf_envelope_plots(
                component_acf_summary,
                component_acf_comparison,
                paths.decomposition_output_dir,
                k=k,
            )
        )
    return outputs


def plot_memo_energy_profile(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
    components: list[str],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    envelope_axis, excess_axis = axes
    x = np.arange(len(components))

    empirical = empirical_by_component(comparison, DETAIL_ENERGY_SHARE, components)
    envelope_axis.plot(
        x,
        empirical,
        marker="o",
        linewidth=1.9,
        color=FINAL_COLOR,
        label="EUR/USD",
    )
    for baseline_type, color, label in [
        ("shuffle", SHUFFLE_COLOR, "Shuffled median + 5-95%"),
        ("gaussian", GAUSSIAN_COLOR, "Gaussian median + 5-95%"),
    ]:
        baseline = summary_values(summary, baseline_type, DETAIL_ENERGY_SHARE, components)
        envelope_axis.fill_between(
            x,
            baseline["p05"],
            baseline["p95"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        envelope_axis.plot(
            x,
            baseline["median"],
            marker="o",
            linewidth=1.4,
            color=color,
            label=label,
        )

        rows = comparison_values(comparison, baseline_type, DETAIL_ENERGY_SHARE, components)
        values = rows["difference_from_median"].to_numpy(dtype=float)
        lower = (
            rows["empirical_value"].to_numpy(dtype=float)
            - rows["baseline_p95"].to_numpy(dtype=float)
        )
        upper = (
            rows["empirical_value"].to_numpy(dtype=float)
            - rows["baseline_p05"].to_numpy(dtype=float)
        )
        excess_axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        excess_axis.plot(
            x,
            values,
            marker="o",
            linewidth=1.7,
            color=color,
            label=f"EUR/USD - {baseline_type} median",
        )

    envelope_axis.set_title("Detail Energy Share")
    envelope_axis.set_ylabel("Share of detail-layer energy")
    envelope_axis.legend()
    envelope_axis.grid(axis="y", alpha=0.25)

    excess_axis.axhline(0.0, color="black", linewidth=0.9, alpha=0.8)
    excess_axis.set_title("Excess vs Baseline Median")
    excess_axis.set_ylabel("EUR/USD minus baseline median")
    excess_axis.legend()
    excess_axis.grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.set_xlabel("Component")
        axis.set_xticks(x)
        axis.set_xticklabels(components, rotation=35)

    fig.suptitle("Volatility Energy Redistribution Across Scales")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_memo_entropy_profile(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
    components: list[str],
) -> Path:
    theoretical_probabilities = np.array(
        [1 / 8, 3 / 16, 3 / 16, 3 / 16, 3 / 16, 1 / 8],
        dtype=float,
    )
    theoretical_entropy = float(
        -np.sum(theoretical_probabilities * np.log(theoretical_probabilities))
    )
    theoretical_normalized_entropy = theoretical_entropy / math.log(6)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(components))
    empirical = empirical_by_component(comparison, NORMALIZED_ENTROPY, components)
    ax.plot(x, empirical, marker="o", linewidth=1.9, color=FINAL_COLOR, label="EUR/USD")

    for baseline_type, color, label in [
        ("shuffle", SHUFFLE_COLOR, "Shuffled median + 5-95%"),
        ("gaussian", GAUSSIAN_COLOR, "Gaussian median + 5-95%"),
    ]:
        baseline = summary_values(summary, baseline_type, NORMALIZED_ENTROPY, components)
        ax.fill_between(
            x,
            baseline["p05"],
            baseline["p95"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            x,
            baseline["median"],
            marker="o",
            linewidth=1.4,
            color=color,
            label=label,
        )

    ax.axhline(
        theoretical_normalized_entropy,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label=f"Theoretical reference ({theoretical_normalized_entropy:.4f})",
    )
    ax.set_title("Normalized Permutation Entropy Across Scales")
    ax.set_xlabel("Component")
    ax.set_ylabel("Normalized entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_metric_envelope(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
    metric: str,
    components: list[str],
    title: str,
    ylabel: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(components))
    empirical = empirical_by_component(comparison, metric, components)
    ax.plot(x, empirical, marker="o", linewidth=1.9, color=FINAL_COLOR, label="EUR/USD")

    for baseline_type, color, label in [
        ("shuffle", SHUFFLE_COLOR, "Shuffled median + 5-95%"),
        ("gaussian", GAUSSIAN_COLOR, "Gaussian median + 5-95%"),
    ]:
        baseline = summary_values(summary, baseline_type, metric, components)
        ax.fill_between(
            x,
            baseline["p05"],
            baseline["p95"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            x,
            baseline["median"],
            marker="o",
            linewidth=1.4,
            color=color,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_metric_excess(
    comparison: pd.DataFrame,
    output_path: Path,
    metric: str,
    components: list[str],
    title: str,
    ylabel: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(components))
    for baseline_type, color, label in [
        ("shuffle", SHUFFLE_COLOR, "EUR/USD - shuffled median"),
        ("gaussian", GAUSSIAN_COLOR, "EUR/USD - Gaussian median"),
    ]:
        rows = comparison_values(comparison, baseline_type, metric, components)
        values = rows["difference_from_median"].to_numpy(dtype=float)
        lower = (
            rows["empirical_value"].to_numpy(dtype=float)
            - rows["baseline_p95"].to_numpy(dtype=float)
        )
        upper = (
            rows["empirical_value"].to_numpy(dtype=float)
            - rows["baseline_p05"].to_numpy(dtype=float)
        )
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(x, values, marker="o", linewidth=1.7, color=color, label=label)
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_acf_envelope(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
    acf_kind: str,
    max_lag: int,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    empirical = comparison[
        (comparison["baseline_type"] == "shuffle")
        & (comparison["acf_kind"] == acf_kind)
        & (comparison["lag"] <= max_lag)
    ].sort_values("lag")
    lags = empirical["lag"].to_numpy()
    ax.plot(
        lags,
        empirical["empirical_value"].to_numpy(dtype=float),
        color=FINAL_COLOR,
        linewidth=1.2,
        label="EUR/USD",
    )
    for baseline_type, color, label in [
        ("shuffle", SHUFFLE_COLOR, "Shuffled median + 5-95%"),
        ("gaussian", GAUSSIAN_COLOR, "Gaussian median + 5-95%"),
    ]:
        rows = summary[
            (summary["baseline_type"] == baseline_type)
            & (summary["acf_kind"] == acf_kind)
            & (summary["lag"] <= max_lag)
            & (summary["metric"] == "acf")
        ].sort_values("lag")
        ax.fill_between(
            rows["lag"].to_numpy(),
            rows["p05"].to_numpy(dtype=float),
            rows["p95"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            rows["lag"].to_numpy(),
            rows["median"].to_numpy(dtype=float),
            color=color,
            linewidth=1.0,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_xlim(1, max_lag)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def create_component_acf_envelope_plots(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
    k: int,
) -> list[Path]:
    short_layers = [f"D_{scale:02d}" for scale in range(1, min(k, 6) + 1)]
    long_layers = [f"D_{scale:02d}" for scale in range(7, k + 1)] + [f"A_{k:02d}"]
    return [
        plot_component_acf_grid_envelope(
            summary,
            comparison,
            output_dir / "layer_acf_returns_short_scales.png",
            layers=short_layers,
            acf_kind="component",
            title="Short-Scale Layer Autocorrelation",
        ),
        plot_component_acf_grid_envelope(
            summary,
            comparison,
            output_dir / "layer_acf_abs_returns_short_scales.png",
            layers=short_layers,
            acf_kind="absolute_component",
            title="Short-Scale Absolute Layer Autocorrelation",
        ),
        plot_component_acf_grid_envelope(
            summary,
            comparison,
            output_dir / "layer_acf_returns_long_scales.png",
            layers=long_layers,
            acf_kind="component",
            title="Long-Scale Layer Autocorrelation",
        ),
        plot_component_acf_grid_envelope(
            summary,
            comparison,
            output_dir / "layer_acf_abs_returns_long_scales.png",
            layers=long_layers,
            acf_kind="absolute_component",
            title="Long-Scale Absolute Layer Autocorrelation",
        ),
    ]


def plot_component_acf_grid_envelope(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
    layers: list[str],
    acf_kind: str,
    title: str,
) -> Path:
    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(16, 22),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(f"{title} with Monte Carlo Baseline Envelopes", fontsize=16)

    for axis, layer in zip(axes, layers, strict=True):
        empirical = comparison[
            (comparison[COMPONENT] == layer)
            & (comparison["baseline_type"] == "shuffle")
            & (comparison["acf_kind"] == acf_kind)
            & (comparison["metric"] == "acf")
        ].sort_values("lag")
        axis.plot(
            empirical["lag"].to_numpy(),
            empirical["empirical_value"].to_numpy(dtype=float),
            linewidth=1.0,
            color=FINAL_COLOR,
            label="EUR/USD",
        )

        for baseline_type, color, label in [
            ("shuffle", SHUFFLE_COLOR, "Shuffled median + 5-95%"),
            ("gaussian", GAUSSIAN_COLOR, "Gaussian median + 5-95%"),
        ]:
            rows = summary[
                (summary[COMPONENT] == layer)
                & (summary["baseline_type"] == baseline_type)
                & (summary["acf_kind"] == acf_kind)
                & (summary["metric"] == "acf")
            ].sort_values("lag")
            axis.fill_between(
                rows["lag"].to_numpy(),
                rows["p05"].to_numpy(dtype=float),
                rows["p95"].to_numpy(dtype=float),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                rows["lag"].to_numpy(),
                rows["median"].to_numpy(dtype=float),
                linewidth=0.9,
                color=color,
                label=label,
            )

        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.75)
        axis.set_ylabel(layer)
        axis.grid(axis="y", alpha=0.2)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Lag in original 5m observations")
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_correlation_memo(
    final_decomposition_csv: Path,
    comparison: pd.DataFrame,
    output_path: Path,
    k: int,
) -> Path:
    components = decomposition_components(k, include_original=False)
    final_corr = empirical_abs_corr(final_decomposition_csv, components)
    shuffle = corr_matrix(comparison, "shuffle", "difference_from_median", components)
    outside = corr_matrix(comparison, "shuffle", "outside_envelope", components)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_heatmap(fig, axes[0], final_corr, components, "EUR/USD Absolute Component Correlation", 0, 1, "viridis")
    limit = max(0.05, float(np.nanmax(np.abs(shuffle))))
    draw_heatmap(fig, axes[1], shuffle, components, "EUR/USD - Shuffled Median", -limit, limit, "coolwarm")
    draw_heatmap(
        fig,
        axes[2],
        outside,
        components,
        "Outside Shuffled Envelope (1 = outside 5-95%)",
        0,
        1,
        "Reds",
        colorbar_label="Outside envelope",
    )
    fig.suptitle("Cross-Scale Volatility Coupling")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def plot_correlation_result_plots(
    final_decomposition_csv: Path,
    comparison: pd.DataFrame,
    output_dir: Path,
    k: int,
) -> list[Path]:
    components = decomposition_components(k, include_original=False)
    final_corr = empirical_abs_corr(final_decomposition_csv, components)
    outputs: list[Path] = []
    outputs.append(save_heatmap(final_corr, components, output_dir / "abs_corr_empirical.png", "EUR/USD Absolute Component Correlation", 0, 1, "viridis"))
    for baseline_type in ["shuffle", "gaussian"]:
        difference = corr_matrix(comparison, baseline_type, "difference_from_median", components)
        outside = corr_matrix(comparison, baseline_type, "outside_envelope", components)
        limit = max(0.05, float(np.nanmax(np.abs(difference))))
        outputs.append(save_heatmap(difference, components, output_dir / f"abs_corr_empirical_minus_{baseline_type}_median.png", f"EUR/USD - {baseline_type.title()} Median", -limit, limit, "coolwarm"))
        outputs.append(save_heatmap(outside, components, output_dir / f"abs_corr_outside_{baseline_type}_envelope.png", f"Outside {baseline_type.title()} Envelope (1 = outside 5-95%)", 0, 1, "Reds", colorbar_label="Outside envelope"))
    return outputs


def empirical_by_component(
    comparison: pd.DataFrame,
    metric: str,
    components: list[str],
) -> np.ndarray:
    rows = comparison_values(comparison, "shuffle", metric, components)
    return rows["empirical_value"].to_numpy(dtype=float)


def summary_values(
    summary: pd.DataFrame,
    baseline_type: str,
    metric: str,
    components: list[str],
) -> pd.DataFrame:
    rows = (
        summary[(summary["baseline_type"] == baseline_type) & (summary["metric"] == metric)]
        .set_index(COMPONENT)
        .reindex(components)
    )
    if rows[["median", "p05", "p95"]].isna().any().any():
        missing = rows[rows[["median", "p05", "p95"]].isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing summary values for {metric}: {missing}")
    return rows


def comparison_values(
    comparison: pd.DataFrame,
    baseline_type: str,
    metric: str,
    components: list[str],
) -> pd.DataFrame:
    rows = (
        comparison[
            (comparison["baseline_type"] == baseline_type)
            & (comparison["metric"] == metric)
        ]
        .set_index(COMPONENT)
        .reindex(components)
    )
    if rows["empirical_value"].isna().any():
        missing = rows[rows["empirical_value"].isna()].index.tolist()
        raise ValueError(f"Missing comparison values for {metric}: {missing}")
    return rows


def empirical_abs_corr(final_decomposition_csv: Path, components: list[str]) -> np.ndarray:
    frame = pd.read_csv(final_decomposition_csv, usecols=components)
    return absolute_component_correlation(frame, components).to_numpy()


def corr_matrix(
    comparison: pd.DataFrame,
    baseline_type: str,
    value_column: str,
    components: list[str],
) -> np.ndarray:
    rows = comparison[
        (comparison["baseline_type"] == baseline_type)
        & (comparison["metric"] == "correlation_abs")
    ].copy()
    if value_column == "outside_envelope":
        rows[value_column] = ~rows["inside_envelope"].astype(bool)
    matrix = rows.pivot(index="component_i", columns="component_j", values=value_column)
    return matrix.reindex(index=components, columns=components).to_numpy(dtype=float)


def save_heatmap(
    values: np.ndarray,
    components: list[str],
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8.5))
    draw_heatmap(fig, ax, values, components, title, vmin, vmax, cmap, colorbar_label=colorbar_label)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path


def draw_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    values: np.ndarray,
    components: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str = "",
) -> None:
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticklabels(components)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=8)
