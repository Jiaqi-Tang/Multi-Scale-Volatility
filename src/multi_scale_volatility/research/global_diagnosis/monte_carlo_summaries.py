"""Summary and empirical-comparison helpers for Monte Carlo metrics."""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from multi_scale_volatility.core.components import component_specs, decomposition_components
from multi_scale_volatility.core.config.names import (
    ANNUALIZED_RMS_VOLATILITY,
    BASE_INTERVAL_MINUTES,
    COMPONENT,
    COMPONENT_TYPE,
    DETAIL_ENERGY_SHARE,
    EFFECTIVE_N,
    ENERGY,
    K,
    LOG_RETURN,
    MONTE_CARLO_BASELINE_QUANTILE_METHOD,
    NORMALIZED_ENTROPY,
    ORDINAL_WINDOWS,
    PERMUTATION_ENTROPY,
    REPEAT_LENGTH,
    RMS_VOLATILITY,
    SCALE_DAYS,
    SCALE_MINUTES,
    SERIES_FINAL,
    TOTAL_COMPONENT_ENERGY_SHARE,
)
from multi_scale_volatility.core.stats import (
    absolute_component_correlation,
    autocorrelation,
    compressed_layer_autocorrelation,
)
from multi_scale_volatility.core.utils.validation import require_finite_array
from multi_scale_volatility.research.global_diagnosis.monte_carlo_rows import (
    compute_component_acf_rows,
    compute_entropy_rows,
    compute_volatility_rows,
)

SUMMARY_QUANTILES = (0.05, 0.5, 0.95)

def summarize_metrics(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_cols, group_key, strict=True))
        for metric in value_cols:
            values = group[metric].dropna().astype(float).to_numpy()
            if len(values) == 0:
                continue
            p05, median, p95 = np.quantile(
                values,
                SUMMARY_QUANTILES,
                method=MONTE_CARLO_BASELINE_QUANTILE_METHOD,
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_simulations": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(median),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "p05": float(p05),
                    "p95": float(p95),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "quantile_method": MONTE_CARLO_BASELINE_QUANTILE_METHOD,
                }
            )
    return pd.DataFrame(rows)


def compare_empirical_volatility(
    empirical_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    empirical = pd.read_csv(empirical_csv)
    empirical = empirical[empirical["series"] == SERIES_FINAL].copy()
    value_cols = [
        ENERGY,
        RMS_VOLATILITY,
        ANNUALIZED_RMS_VOLATILITY,
        DETAIL_ENERGY_SHARE,
        TOTAL_COMPONENT_ENERGY_SHARE,
    ]
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=[
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
        ],
        value_cols=value_cols,
    )


def compare_empirical_entropy(
    empirical_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    empirical = pd.read_csv(empirical_csv)
    empirical = empirical[empirical["series"] == SERIES_FINAL].copy()
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=[
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
            REPEAT_LENGTH,
        ],
        value_cols=[
            EFFECTIVE_N,
            ORDINAL_WINDOWS,
            PERMUTATION_ENTROPY,
            NORMALIZED_ENTROPY,
        ],
    )


def compare_empirical_acf(
    final_returns_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    return_acf_max_lag: int,
    abs_return_acf_max_lag: int,
) -> pd.DataFrame:
    returns = pd.read_csv(final_returns_csv, usecols=[LOG_RETURN])
    values = returns[LOG_RETURN].astype(float).to_numpy()
    empirical = pd.DataFrame(
        [
            *(
                {
                    "acf_kind": "return",
                    "lag": lag,
                    "max_lag": return_acf_max_lag,
                    "acf": float(value),
                }
                for lag, value in enumerate(
                    autocorrelation(values, return_acf_max_lag),
                    start=1,
                )
            ),
            *(
                {
                    "acf_kind": "absolute_return",
                    "lag": lag,
                    "max_lag": abs_return_acf_max_lag,
                    "acf": float(value),
                }
                for lag, value in enumerate(
                    autocorrelation(np.abs(values), abs_return_acf_max_lag),
                    start=1,
                )
            ),
        ]
    )
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=["acf_kind", "lag", "max_lag"],
        value_cols=["acf"],
    )


def compare_empirical_component_acf(
    final_decomposition_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
    short_max_lag: int,
    long_max_lag: int,
) -> pd.DataFrame:
    components = decomposition_components(k, include_original=False)
    frame = pd.read_csv(final_decomposition_csv, usecols=components)
    empirical = pd.DataFrame(
        compute_component_acf_rows(
            frame,
            baseline_type=SERIES_FINAL,
            simulation_id=0,
            k=k,
            short_max_lag=short_max_lag,
            long_max_lag=long_max_lag,
        )
    )
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=[
            COMPONENT,
            K,
            COMPONENT_TYPE,
            SCALE_MINUTES,
            SCALE_DAYS,
            "acf_kind",
            "lag",
            "compressed_lag",
            "max_lag",
        ],
        value_cols=["acf"],
    )


def compare_empirical_abs_component_correlation(
    final_decomposition_csv: Path,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    components = decomposition_components(k, include_original=False)
    frame = pd.read_csv(final_decomposition_csv, usecols=components)
    corr = absolute_component_correlation(frame, components)
    empirical = pd.DataFrame(
        [
            {
                "component_i": component_i,
                "component_j": component_j,
                "correlation_abs": float(corr.loc[component_i, component_j]),
            }
            for component_i, component_j in itertools.product(components, components)
        ]
    )
    return compare_empirical_metric_table(
        empirical,
        simulations,
        summary,
        index_cols=["component_i", "component_j"],
        value_cols=["correlation_abs"],
    )


def compare_empirical_metric_table(
    empirical: pd.DataFrame,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
    index_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    empirical = normalize_comparison_keys(empirical, index_cols)
    simulations = normalize_comparison_keys(simulations, index_cols)
    summary = normalize_comparison_keys(summary, index_cols)
    empirical_long = empirical.melt(
        id_vars=index_cols,
        value_vars=value_cols,
        var_name="metric",
        value_name="empirical_value",
    ).dropna(subset=["empirical_value"])
    simulations_long = simulations.melt(
        id_vars=["baseline_type", "simulation_id", *index_cols],
        value_vars=value_cols,
        var_name="metric",
        value_name="simulated_value",
    ).dropna(subset=["simulated_value"])

    keys = ["baseline_type", *index_cols, "metric"]
    comparison = summary.merge(
        empirical_long,
        on=[*index_cols, "metric"],
        how="inner",
    )
    joined = simulations_long.merge(
        comparison[[*keys, "empirical_value"]],
        on=keys,
        how="inner",
    )
    ranks = (
        joined.assign(le_empirical=joined["simulated_value"] <= joined["empirical_value"])
        .groupby(keys, dropna=False)
        .agg(
            percentile_rank=("le_empirical", "mean"),
            n_simulations=("simulated_value", "count"),
        )
        .reset_index()
    )

    output = comparison.merge(ranks, on=keys, how="inner")
    if "n_simulations_y" in output.columns:
        output["n_simulations"] = output["n_simulations_y"]
        output = output.drop(
            columns=[
                column
                for column in ["n_simulations_x", "n_simulations_y"]
                if column in output.columns
            ]
        )
    output = output.rename(
        columns={
            "median": "baseline_median",
            "p05": "baseline_p05",
            "p95": "baseline_p95",
        }
    )
    output["difference_from_median"] = (
        output["empirical_value"] - output["baseline_median"]
    )
    output["inside_envelope"] = (
        (output["baseline_p05"] <= output["empirical_value"])
        & (output["empirical_value"] <= output["baseline_p95"])
    )
    output["above_envelope"] = output["empirical_value"] > output["baseline_p95"]
    output["below_envelope"] = output["empirical_value"] < output["baseline_p05"]
    output["quantile_method"] = MONTE_CARLO_BASELINE_QUANTILE_METHOD

    columns = [
        "baseline_type",
        *index_cols,
        "metric",
        "empirical_value",
        "baseline_median",
        "baseline_p05",
        "baseline_p95",
        "difference_from_median",
        "percentile_rank",
        "inside_envelope",
        "above_envelope",
        "below_envelope",
        "n_simulations",
        "quantile_method",
    ]
    return output[columns].sort_values(["baseline_type", *index_cols, "metric"])


def normalize_comparison_keys(frame: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in index_cols:
        if column in output.columns and pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].round(12)
    return output

