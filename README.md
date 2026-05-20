# Multi-Scale Volatility Structure in EUR/USD Returns

Currently in development.

- See `plots/` or `results/` for intermediate results and findings.
- See `Documentation.md` for exact research design and methodology.

## Objective

Quantify how **volatility energy** and **permutation entropy** are distributed across temporal scales in EUR/USD returns using a simple, reversible block-decomposition framework.

This baseline is intentionally minimalist:

- No forecasting
- No rolling windows
- No regime classification
- No event studies
- No optimization-heavy methods

Core question:

> How do volatility and ordering structure change as return information is progressively compressed across sub-hourly, multi-hour, near-daily, and multi-day scales?

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
```
