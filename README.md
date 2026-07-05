# Multi-Scale Volatility Structure in EUR/USD Returns

Current version: **V2.3**. The V1.1 global Monte Carlo baseline pipeline is retained.

V2.3 adds rolling volatility-state regime diagnostics on top of the V2.1 rolling
metrics and rolling baseline correlation envelopes.

- `Memo.md` - concise research summary and findings
- `Documentation.md` - exact preprocessing, decomposition, metric, rolling, and baseline-envelope definitions
- `README.md` - project overview and reproduction guide
- `plots/` and `results/` - generated figures, metric tables, and comparisons

## Objective

This project explores the **multi-scale** structure of **EUR/USD volatility** using a minimalist dyadic decomposition framework applied to 5-minute log returns.

The analysis compares real EUR/USD returns against two reference processes:

- shuffled-return baselines, which preserve the empirical return distribution
  while destroying temporal order
- variance-matched Gaussian baselines, which represent independent Gaussian
  increments with the empirical population variance

The primary goal is to identify whether real FX volatility exhibits
scale-dependent structure beyond heavy tails or independent noise alone.

The global V1.1 analysis is a minimalist full-sample exploration. V2.1 adds
rolling windows to inspect time-local volatility structure. V2.3 adds
exploratory regime-style diagnostics. The project still does not include
forecasting, event studies, formal regime modeling, or trading rules.

## Key Findings

- EUR/USD volatility exhibits excess finest-scale energy relative to the median
  shuffled and Gaussian baseline profiles.
- Intermediate decomposition scales show relative energy deficits.
- Volatility states exhibit persistent cross-scale coupling beyond what is
  explained by the shuffled baseline envelope.
- Rolling windows expose time-local shifts in fine, mid, and coarse volatility
  share structure.
- Rolling regime maps separate volatility level from scale-composition state.
- Absolute-return autocorrelation confirms strong volatility clustering.
- Permutation entropy differences remain comparatively weak under the current
  specification.

### Example Decomposition

![Decomposition](plots/memo/figure_01_decomposition_example.png)

### Cross-Scale Volatility Coupling

![Cross Scale Correlation](plots/memo/figure_05_cross_scale_correlation.png)

## Reproduce the Pipeline

Ensure that Python 3.13 is installed.

Install dependencies:

```powershell
pip install -e .
```

Run the full V1.1 global pipeline (runtime around `14min`):

```powershell
ve run-all
```

This runs preprocessing, length standardization, empirical decomposition,
empirical metrics, Monte Carlo baseline generation, Monte Carlo baseline metrics
and comparisons, and V1.1 plot generation.

Or run each V1.1 step explicitly:

```powershell
ve preprocess
ve standardize
ve decompose
ve volatility
ve entropy
ve baselines
ve monte-carlo-metrics
ve monte-carlo-comparisons
ve plot memo
```

Run the V2.1 rolling diagnostics:

```powershell
ve rolling
ve plot rolling
ve plot rolling-examples
```

Run rolling baseline correlation envelopes:

```powershell
ve rolling-baselines
ve plot rolling-baselines
```

Run V2.3 rolling regime diagnostics:

```powershell
ve rolling-regimes
ve plot rolling-regimes
```

## Repository Structure

```text
src/
  multi_scale_volatility/
    app/
    core/
      config/
      io/
      stats/
      utils/
    plotting/
    research/
      preprocessing/
      global_diagnosis/
      rolling_window_diagnosis/

data/
  raw/
  processed/
  derived/
    decomposition/
    monte_carlo_baselines/
      returns/
      decomposition/

results/
  global_diagnosis/
    volatility/
    entropy/
    monte_carlo_baselines/
  rolling_window_diagnosis/
    rolling_metrics/
    rolling_baselines/
    regimes/

plots/
  memo/
  results/
    global_diagnosis/
      data_eda/
      volatility/
      entropy/
      correlation/
    rolling_window_diagnosis/
      rms/
      energy_share/
      examples/
      rolling_baselines/
      regimes/

Documentation.md
Memo.md
README.md
```

## Current Status

V2.3 complete:

- preprocessing pipeline,
- dyadic decomposition,
- Monte Carlo baseline construction,
- volatility diagnostics,
- entropy diagnostics,
- cross-scale correlation analysis,
- fixed-observation rolling decompositions,
- rolling RMS and energy-share diagnostics,
- rolling baseline correlation-envelope comparisons,
- rolling volatility-state regime diagnostics.

Currently exploring:

- event-transition analysis,
- possible V3 extensions.

Known runtime notes: entropy is the slowest global Monte Carlo metric stage; the
rolling baseline run is dominated by rolling metric computation for $W=8192$.
