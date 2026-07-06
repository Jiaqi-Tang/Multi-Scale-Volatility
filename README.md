# Multi-Scale Volatility Structure in EUR/USD Returns

- `Memo.md` - concise research summary and findings
- `Documentation.md` - exact preprocessing, decomposition, metric, rolling, and baseline-envelope definitions
- `README.md` - project overview and reproduction guide
- `plots/` and `results/` - generated figures, metric tables, and comparisons

## Objective

This project studies empirical volatility structure in EUR/USD 5-minute returns using a dyadic multi-scale decomposition. The goal is to separate:

- volatility level,
- scale allocation,
- cross-scale dependence,

and compare these empirical features against Gaussian and shuffled-return baselines.

The motivating question is:

> How is empirical EUR/USD volatility structured across time and scale, and how does this structure differ from variance-matched Gaussian noise and shuffled heavy-tailed returns?

The project is intentionally minimalist: it does not use forecasting models, option data, order book data, or trading rules. Instead, it focuses on transparent volatility diagnostics that can be interpreted across scales and through time.

## Key Findings

- **Rolling RMS volatility synchronizes strongly across scales:** High-volatility periods tend to raise RMS across many detail components simultaneously.
- **Scale composition is comparatively stable:** Fine-scale detail energy dominates through time, while mid and coarse shares fluctuate within narrower ranges.
- **Volatility level and scale composition are distinct.** Total RMS volatility is nearly uncorrelated with fine/mid/coarse energy shares.
- **Global energy allocation differs from baselines:** Empirical returns show excess finest-scale detail energy and intermediate-scale energy deficits relative to both Gaussian and shuffled baselines.

### Global Energy Allocation

![Global energy profile](plots/memo/figure_02_energy_profile.png)

### Rolling RMS Synchronization

![Rolling RMS volatility structure](plots/memo/figure_04_rolling_rms_structure.png)

## Version Roadmap

| Version    | Focus                                                                               | Status   |
| ---------- | ----------------------------------------------------------------------------------- | -------- |
| V1.0       | Initial full-sample dyadic decomposition and diagnostics                            | Complete |
| V1.1       | Monte Carlo Gaussian/shuffled baselines with median and 5–95% envelopes             | Complete |
| V2.1       | Rolling-window RMS volatility and energy-share diagnostics                          | Complete |
| V2.2       | Rolling baseline envelopes for cross-scale correlation diagnostics                  | Complete |
| V2.3       | Exploratory volatility-state regime maps using total RMS and fine-share percentiles | Complete |
| V3 planned | Event-aligned transition analysis of high-volatility episodes                       | Planned  |

## Reproduce the Pipeline

Ensure that Python 3.13 is installed.

Install dependencies:

```powershell
pip install -e .
```

Run the full current pipeline:

```powershell
ve run-all
```

Starting from the source code and `data/raw/`, this regenerates the derived
`data/`, `results/`, and `plots/` artifacts for the current analysis.

The explicit equivalent is:

```powershell
# data processing
ve preprocess
ve standardize

# global analysis
ve decompose
ve volatility
ve entropy

# Monte Carlo baselines
ve baselines
ve monte-carlo-metrics

# rolling analysis
ve rolling
ve rolling-baselines
ve rolling-regimes

# plotting
ve plot monte-carlo-baselines
ve plot rolling
ve plot rolling-examples
ve plot rolling-baselines
ve plot rolling-regimes
ve plot memo
```

## Repository Structure

```text
src/
  multi_scale_volatility/
    app/
    core/
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
    rolling_window_diagnosis/

Documentation.md
Memo.md
README.md
```

## Current Status and Next Steps

V2.3 is complete. The current results suggest that EUR/USD volatility level is strongly synchronized across decomposition scales, while fine/mid/coarse energy-share composition is comparatively stable and mostly independent of total RMS volatility.

The natural V3 direction is **event-aligned transition analysis**. The goal is to test whether high-volatility episodes begin as fine-scale bursts and then spread into broader mid-scale activation, or whether multiple scales activate simultaneously.
