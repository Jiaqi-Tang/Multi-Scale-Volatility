# Multi-Scale Volatility Structure in EUR/USD Returns

Baseline version complete; extensions and optimizations ongoing.

- `Memo.md` - concise research summary and findings
- `Documentation.md` - exact preprocessing, decomposition, and metric definitions
- `README.md` - project overview and reproduction guide
- `plots/` and `results/` - generated figures, metric tables, and findings

## Objective

This project explores the **multi-scale** structure of **EUR/USD volatility** using a minimalist dyadic decomposition framework applied to 5-minute log returns. The analysis compares real EUR/USD returns against two reference processes: a shuffled-return baseline, and a variance-matched Gaussian baseline.

The primary goal is to identify whether real FX volatility exhibits scale-dependent structure beyond heavy tails or independent noise alone.

This current project is intended as a **minimalist first-stage exploration** of multi-scale volatility structure. The design intentionally has

- No forecasting
- No rolling windows
- No regime classification
- No event studies
- No optimization-heavy methods

## Key Findings

- EUR/USD volatility exhibits excess finest-scale energy relative to shuffled and Gaussian baselines.
- Intermediate decomposition scales show relative energy deficits.
- Volatility states exhibit persistent cross-scale coupling beyond what is explained by heavy tails alone.
- Absolute-return autocorrelation confirms strong volatility clustering.
- Permutation entropy differences were comparatively weak under the current specification.

### Example Decomposition

![Decomposition](plots/memo/figure_01_decomposition_example.png)

### Cross-Scale Volatility Coupling

![Cross Scale Correlation](plots/memo/figure_05_cross_scale_correlation.png)

## Reproduce the Pipeline

Ensure that Python 3.13 is installed

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python scripts/run_all.py
```

Or run each step explicitly:

```powershell
python scripts/run_preprocessing.py
python scripts/run_length_standardization.py
python scripts/run_baselines.py
python scripts/run_decomposition.py
python scripts/run_volatility.py
python scripts/run_entropy.py
python scripts/run_eda_plots.py
python scripts/run_decomposition_plots.py
python scripts/run_volatility_plots.py
python scripts/run_entropy_plots.py
python scripts/run_memo_plots.py
```

## Repository Structure

```text
data/
  raw/
  intermediate/
  final/
  baselines/
  decomposition/

results/
  volatility/
  entropy/

plots/
  eda/
  results/
  memo/

scripts/
  run_preprocessing.py
  run_length_standardization.py
  run_baselines.py
  run_decomposition.py
  run_volatility.py
  run_entropy.py
  run_eda_plots.py
  run_decomposition_plots.py
  run_volatility_plots.py
  run_entropy_plots.py
  run_memo_plots.py
  run_all.py

Documentation.md
Memo.md
README.md
```

## Current Status

V1 complete:

- preprocessing pipeline,
- dyadic decomposition,
- baseline construction,
- volatility diagnostics,
- entropy diagnostics,
- cross-scale correlation analysis.

Currently exploring:

- time-local volatility propagation,
- event-transition analysis,
- robustness check on more baselines.
