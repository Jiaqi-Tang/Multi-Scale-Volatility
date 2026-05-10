# Multi-Scale Volatility–Entropy Decomposition of EUR/USD Returns

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
