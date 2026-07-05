# Data Source

Raw data source: [HistData's MetaTrader EUR/USD 1-minute bar data](https://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes/EURUSD).

Local raw files are stored under `data/raw/metatrader`.

Asset: EUR/USD.

Raw frequency: 1-minute OHLC bars.

Raw files used: calendar years 2016 through 2025.

Observed cleaned timestamp range: 2016-01-03 22:00 UTC to 2025-12-31 21:57 UTC.

Raw fields: timestamp, open bid, high bid, low bid, close bid, volume.

## Raw Observations

Let the raw 1-minute price bars be indexed by observed timestamps:

$$
t \in \mathcal{T}_{1m}
$$

For each timestamp $t$, the raw observation contains:

$$
(O_t, H_t, L_t, C_t, V_t)
$$

The data source states that all timestamps use Eastern Standard Time without DST. This project
interprets raw timestamps as fixed UTC-05:00, with no daylight-saving adjustment.
All analysis timestamps are converted to UTC.

The raw files contain missing observations. Missingness is not repaired by
forward-filling or interpolation.

## Caveats

These are treated as vendor data artifacts.

**Zeros in Volume field**

The volume field $V_t$ is not used, as $V_t=0$ for all raw observations.

**Daylight saving time**

HistData's files are intended to be fixed EST, but from 2019 onward some files show
an EU daylight-saving-time transition artifact around 19:00 file time.

Observed issues:

- EU DST-end duplicate rows appear from 19:00 through 19:59 in 2019, 2020, 2021,
  2022, 2023, and 2025.
- These duplicate rows are exact duplicates, so they are removed by exact-row
  deduplication.
- EU DST-start has a missing 19:00 through 19:59 hour from 2019 onward.
- 2024 has the EU DST-start missing-hour pattern but does not have the corresponding
  EU DST-end duplicate-hour pattern.

---

# Preprocessing

Objective: Transform raw 1-minute EUR/USD OHLC data into a clean 5-minute log-return series suitable for volatility and entropy analysis.

## Data cleaning

**Load Raw Data**

Load all raw EUR/USD MetaTrader CSV files from 2016 through 2025.

The initial raw dataset is:

$$
\mathcal{X}_{1m}^{raw} = \{(t_i, O_i, H_i, L_i, C_i, V_i)\}_{i=1}^{N_{raw}}
$$

**Timestamp Interpretation**

Each raw timestamp is converted to UTC:

$$
t_i^{UTC} = t_i^{raw} + 5\text{ hours}
$$

The preprocessing does not apply daylight-saving-time shifts.

**Exact Deduplication**

An exact duplicate means the full raw observation is repeated:

$$
(t_i, O_i, H_i, L_i, C_i, V_i) = (t_j, O_j, H_j, L_j, C_j, V_j)
$$

for $i \neq j$.

If duplicate timestamps with different OHLC or volume values are found, this is treated as a data-quality error (rather than choosing one observation arbitrarily). No such duplications were found.

The cleaned 1-minute dataset is:

$$
\mathcal{X}_{1m}^{clean} = \{(t_i, O_i, H_i, L_i, C_i, V_i)\}_{i=1}^{N_{1m}}
$$

where timestamps are in UTC.

## Data aggregations

**Aggregate to 5-Minute OHLC Bars**

Let $B_j$ be the set of observed 1-minute bars whose timestamps fall inside the
5-minute interval indexed by $j$.

For every nonempty 5-minute interval:

$$
|B_j| \in \{1,2,3,4,5\}
$$

the 5-minute OHLC bar is defined as:

$$
O_j^{5m} = \text{first observed } O_i \text{ in } B_j
$$

$$
H_j^{5m} = \max_{i \in B_j} H_i
$$

$$
L_j^{5m} = \min_{i \in B_j} L_i
$$

$$
C_j^{5m} = \text{last observed } C_i \text{ in } B_j
$$

The number of observed 1-minute bars used in each 5-minute bar is:

$$
n_j^{1m} = |B_j|
$$

Every nonempty 5-minute bar is kept, including bars constructed from only 1, 2, 3,
or 4 observed 1-minute bars. Empty 5-minute bars are dropped.

The resulting 5-minute OHLC dataset is:

$$
\mathcal{X}_{5m} = \{(t_j, O_j^{5m}, H_j^{5m}, L_j^{5m}, C_j^{5m}, n_j^{1m})\}_{j=1}^{N_{5m}}
$$

**Compute Log Returns**

Let the observed 5-minute close price be:

$$
S_j = C_j^{5m},\quad \text{ where } S_j > 0
$$

For consecutive observed 5-minute timestamps, compute:

$$
r_j = \log(S_j) - \log(S_{j-1})
$$

The elapsed time for each candidate return is:

$$
\Delta t_j = t_j - t_{j-1}
$$

**Gap Filtering**

The expected gap is:

$$
\Delta t_{expected} = 5\text{ minutes}
$$

The final clean return series keeps only returns satisfying:

$$
\Delta t_j = \Delta t_{expected} = 5\text{ minutes}
$$

This strict rule avoids including weekend gaps, holiday gaps, outages, and
missing-candle jumps as ordinary 5-minute returns.

The final clean return series is:

$$
R = \{r_1, r_2, \ldots, r_N\}
$$

of consecutive log returns on 5m data.

---

## Output datasets

Preprocessing outputs are intermediate datasets and can be found in the
`data/processed` folder:

```text
data/processed/eurusd_1m_utc_clean.csv
data/processed/eurusd_5m_ohlc_utc_nonempty.csv
data/processed/eurusd_5m_log_returns_clean.csv
data/processed/preprocessing_report.json
```

The clean 5-minute return dataset contains `timestamp_utc, close, log_return, previous_timestamp_utc, previous_close, delta_minutes, n_m1`, where `n_m1` records the number of observed 1-minute bars used to construct the current 5-minute close bar.

Dropped returns are not exported as a separate dataset. They are recorded only for
debugging and audit purposes in `data/processed/preprocessing_report.json`.

The final analysis dataset will be produced after length standardization and saved
as:

```text
data/processed/eurusd_5m_log_returns_final.csv
```

**Preprocessing Results**

Current preprocessing results:

```text
raw_rows_loaded: 3,671,254
raw_exact_duplicate_rows_dropped: 360
clean_1m_rows: 3,670,894
ohlc_5m_nonempty_rows: 737,034
ohlc_5m_complete_rows: 726,627
ohlc_5m_partial_rows: 10,407
return_rows_clean: 735,706
return_rows_dropped: 1,327
```

Distribution of observed 1-minute bars per retained 5-minute OHLC bar:

```text
1 observed 1-minute bar: 316
2 observed 1-minute bars: 637
3 observed 1-minute bars: 1,647
4 observed 1-minute bars: 7,807
5 observed 1-minute bars: 726,627
```

---

# Length Standardization

Objective: truncate the dataset so its length is divisible by $2^K$, such that Block-Average Multi-Scale Decomposition can be done.

## Design choices

Choose maximum decomposition depth: $K = 11$

This gives block size of $2^K = 2048$.

Since the base return frequency is 5 minutes, the time span of one maximum-depth
block is:

$$
T_K = 5 \times 2^{11} = 10240\text{ minutes} \approx 7.11\text{ days}
$$

The standardized length is:

$$
N^{\ast} = \max \{2^K \cdot m : 2^K \cdot m \leq N\} = 2^K \left\lfloor \frac{N}{2^K} \right\rfloor
$$

With $K=11$ and $N=735,706$:

$$
N^{\ast} = 735{,}232
$$

The final analysis return series is:

$$
R^{\ast} = \{r_1, r_2, \ldots, r_{N^{\ast}}\}
$$

Rows are truncated from the end of the dataset only. The start of the sample is
preserved.

## Results

Rows dropped by truncation:

$$
N - N^{\ast} = 474
$$

Dropped tail timestamp range: `2025-12-30 06:30 UTC to 2025-12-31 21:55 UTC`

Final standardized timestamp range: `2016-01-03 22:05 UTC to 2025-12-30 06:25 UTC`

The standardized final dataset is saved as: `data/processed/eurusd_5m_log_returns_final.csv`

The truncation report is saved as: `data/processed/truncation_report.json`

For $R^{\ast}$:

```text
mean_log_return: 1.3814454282268642e-07
variance_log_return: 8.257150232019612e-08
std_log_return: 0.00028735257493225306
min_log_return: -0.0097635255221106
max_log_return: 0.0126936410859073
median_log_return: 0.0
skewness_log_return: 0.15778024187331052
kurtosis_log_return: 44.2160995969573
```

---

# Monte Carlo Baseline Series

Objective: Create baseline time series so empirical EUR/USD diagnostics can be
interpreted against baseline distributions rather than against one random draw.

## Design choices

Baseline series are generated from the standardized final return series $R^{\ast}$.

V1.1 uses:

```text
100 shuffled simulations
100 Gaussian simulations
```

All baseline simulations have length $|R^{baseline}| = N^{\ast}$, and use the
same timestamp index as $R^{\ast}$.

The timestamps are retained as alignment metadata; the baseline computations are conducted on ordered return index:

$$
i = 1,2,\ldots,N^{\ast}
$$

The Monte Carlo configuration is fixed in code:

```text
n_simulations: 100 per baseline type
baseline_types: shuffle, gaussian
master_seed: 20260609
quantiles: 0.05, 0.50, 0.95
quantile method: linear
```

For reproducibility, every simulation receives a deterministic child seed
derived from the master seed, baseline type, and simulation id.

## Shuffled Baselines

Each shuffled baseline is a random permutation of the standardized returns:

$$
R^{shuffle,m} = \pi_m(R^{\ast}), \quad m=0,\ldots,99
$$

where $\pi$ is a random permutation.

Properties preserved:

- same empirical distribution as $R^{\ast}$
- same mean and variance as $R^{\ast}$
- same minimum and maximum as $R^{\ast}$

Property destroyed:

- temporal ordering

Output:

```text
data/derived/monte_carlo_baselines/returns/shuffle/shuffle_sim_000.parquet
...
data/derived/monte_carlo_baselines/returns/shuffle/shuffle_sim_099.parquet
```

## Gaussian Baselines

Each Gaussian baseline is generated as:

$$
R^{BM,m}_i \sim \mathcal{N}(0, \sigma_R^2), \quad m=0,\ldots,99
$$

where:

$$
\sigma_R^2 = \mathrm{Var}(R^{\ast})
$$

The variance is the population variance of the standardized final return series:

$$
\sigma_R^2 = \frac{1}{N^{\ast}}\sum_{i=1}^{N^{\ast}}(r_i - \bar{r})^2
$$

where $\sigma_R^2 = 8.257150232019612 \times 10^{-8}$.

Gaussian simulations are sampled around zero using the empirical population
variance. They are not rescaled after sampling; realized means and variances are
recorded for audit.

Properties targeted:

- same population variance as $R^{\ast}$
- Gaussian independent increments
- zero mean

Output:

```text
data/derived/monte_carlo_baselines/returns/gaussian/gaussian_sim_000.parquet
...
data/derived/monte_carlo_baselines/returns/gaussian/gaussian_sim_099.parquet
```

## Baseline Decompositions

Each baseline simulation is decomposed using the same block-average
decomposition as the empirical series. The decomposition files are saved as
Parquet:

```text
data/derived/monte_carlo_baselines/decomposition/shuffle/shuffle_decomposition_sim_000.parquet
...
data/derived/monte_carlo_baselines/decomposition/gaussian/gaussian_decomposition_sim_099.parquet
```

The baseline audit and configuration files are:

```text
results/global_diagnosis/monte_carlo_baselines/baseline_simulation_audit.csv
results/global_diagnosis/monte_carlo_baselines/monte_carlo_config.json
results/global_diagnosis/monte_carlo_baselines/runtime_log.csv
```

The audit table records baseline type, simulation id, seed, length, realized
mean, realized population variance, realized population standard deviation,
minimum, maximum, output paths, and decomposition reconstruction error.

---

# Block-Average Multi-Scale Decomposition

Objective: Decompose the final return series and each Monte Carlo baseline
simulation into scale-indexed detail layers and a final approximation layer.

## Design choices

The empirical decomposition is applied to:

$$
R^{\ast}
$$

The same decomposition is then applied to every shuffled and Gaussian Monte
Carlo baseline simulation.

For each scale:

$$
k = 1,2,\ldots,K \quad \text{ with }\,K = 11
$$

the block size is:

$$
B_k = 2^k
$$

For reporting and plotting, detail-layer horizons are indexed by the smaller
time scale in the adjacent-scale difference. Thus $D_k$ is labeled with:

$$
T_k^{detail} = 5 \times 2^{k-1}\text{ minutes}
$$

The final approximation layer $A_{11}$ is labeled with its full block horizon:

$$
T_{11}^{approx} = 5 \times 2^{11}\text{ minutes} \approx 7.11\text{ days}
$$

For each series:

$$
A_0 = R
$$

where $R$ denotes the input series being decomposed.

The approximation layer $A_k$ is defined as the block-mean approximation of the
original input series over consecutive non-overlapping blocks of size $B_k$.

For each block:

$$
\mu_j^{(k)} = \frac{1}{B_k}\sum_{i \in \text{block}_j} A_{0,i}
$$

The block mean is expanded back across its block, so $A_k$ has the same length as
the original series.

The detail layer is:

$$
D_k = A_{k-1} - A_k
$$

The reconstruction identity is:

$$
R = A_K + \sum_{k=1}^{K}D_k
$$

The saved decomposition columns are:

```text
index
timestamp_utc
original
D_01
...
D_11
A_11
```

Only $D_1,\ldots,D_{11}$, $A_{11}$, and the original series are saved. Intermediate
approximation layers $A_1,\ldots,A_{10}$ are computed internally but not exported.

The empirical decomposition output is:

```text
data/derived/decomposition/final_decomposition.csv
data/derived/decomposition/decomposition_report.json
```

Monte Carlo decomposition outputs are saved as Parquet under:

```text
data/derived/monte_carlo_baselines/decomposition/shuffle/
data/derived/monte_carlo_baselines/decomposition/gaussian/
```

## Validation

For each decomposed empirical or simulated series, reconstruction error is
computed as:

$$
\epsilon_i = \mathrm{original}_i - \left(A_{11,i} + \sum_{k=1}^{11}D_{k,i}\right)
$$

The decomposition fails if:

$$
\max_i |\epsilon_i| > 10^{-12}
$$

The empirical decomposition reconstructs to machine precision.

```text
final:
  max_abs_reconstruction_error: 3.469446951953614e-18
  mean_abs_reconstruction_error: 2.2820538114180538e-20
```

Monte Carlo reconstruction errors are recorded per simulation in
`results/global_diagnosis/monte_carlo_baselines/baseline_simulation_audit.csv`.

---

# Volatility / Energy Metrics

Objective: Quantify how return variation is distributed across decomposition
components.

## Design choices

Empirical volatility metrics are computed for:

$$
D_1,\ldots,D_{11},A_{11}
$$

for $R^{\ast}$. The same volatility metrics are computed for every Monte Carlo
baseline decomposition and summarized separately in
`results/global_diagnosis/monte_carlo_baselines`.

Let a decomposition component be:

$$
X_c = \{x_{c,1},x_{c,2},\ldots,x_{c,N^{\ast}}\}
$$

where $c$ denotes one of the saved components.

Component energy is:

$$
E_c = \sum_{i=1}^{N^{\ast}}x_{c,i}^2
$$

RMS volatility is:

$$
\sigma_c^{RMS} = \sqrt{\frac{1}{N^{\ast}}E_c}
$$

RMS is used rather than standard deviation because each detail layer is a
zero-sum reconstruction component and the decomposition identity is expressed in
terms of squared component magnitudes. This makes RMS directly comparable to
energy.

Annualized RMS volatility is reported as:

$$
\sigma_{c,ann}^{RMS} = \sigma_c^{RMS}\sqrt{252 \times 24 \times 12}
$$

with assumptions `trading_days_per_year=252`, `trading_hours_per_day=24`, `periods_per_hour=12`, `periods_per_year=72,576`.

Two energy-share definitions are computed.

Detail energy share is defined only for detail layers:

$$
p_k^{detail} = \frac{E(D_k)}{\sum_{j=1}^{11}E(D_j)}
$$

Total component energy share includes the final approximation:

$$
p_c^{total} = \frac{E_c}{\sum_{j=1}^{11}E(D_j) + E(A_{11})}
$$

where:

$$
c \in \{D_1,\ldots,D_{11},A_{11}\}
$$

The mean of every component is recorded in the report for audit purposes.

## Outputs

Volatility outputs are saved under `results/global_diagnosis/volatility`:

```text
results/global_diagnosis/volatility/layer_volatility.csv
results/global_diagnosis/volatility/volatility_report.json
```

The empirical volatility CSV has one row per component:

```text
series
component
k
component_type
scale_minutes
scale_days
energy
rms_volatility
annualized_rms_volatility
detail_energy_share
total_component_energy_share
```

For detail components, `scale_minutes` follows the smaller-time convention:

$$
\text{scale\_minutes}(D_k) = 5 \times 2^{k-1}
$$

For the approximation component:

$$
\text{scale\_minutes}(A_{11}) = 5 \times 2^{11}
$$

## Results

For the final EUR/USD series, most detail-layer energy is concentrated at the
finest scales:

```text
D_01 detail_energy_share: 0.5128464888450901
D_02 detail_energy_share: 0.2491209086554722
D_03 detail_energy_share: 0.1217764904978138
```

The final approximation energy is small relative to total component energy:

```text
A_11 total_component_energy_share: 0.0003843256082527
```

The component energy sum reconstructs the original return energy to numerical
precision:

```text
final energy_reconstruction_gap: 1.7458257062230587e-14
```

---

# Permutation Entropy

Objective: Quantify temporal ordering structure within each decomposition
component.

## Design choices

Empirical permutation entropy is computed for:

$$
D_1,\ldots,D_{11},A_{11}
$$

for $R^{\ast}$. The same entropy calculation is run for every Monte Carlo
baseline decomposition and summarized separately in
`results/global_diagnosis/monte_carlo_baselines`.

The embedding dimension and delay are:

$$
m = 3, \quad \tau = 1
$$

For each component, deterministic repeated block values created by block-mean
expansion are removed before entropy is computed.

Let the compressed component be:

$$
X_c^{comp}
$$

This compression is used only for entropy calculation. It does not alter the
decomposition outputs or volatility metrics.

For entropy only, deterministic jitter is added after compression:

$$
\tilde{x}_{c,i} = x_{c,i}^{comp} + \epsilon_i
$$

where:

$$
\epsilon_i \sim \mathrm{Uniform}(-10^{-10},10^{-10})
$$

Base jitter seed is `314`.

Component-specific deterministic seeds are derived from the base seed, series
name, and component name. The jitter is used only to break ties in ordinal
ranking. It is not used for returns, decomposition, volatility, or energy.

For each ordinal window:

$$
(\tilde{x}_{c,i},\tilde{x}_{c,i+\tau},\tilde{x}_{c,i+2\tau})
$$

the rank-order pattern is counted.

There are $3! = 6$ possible ordinal patterns. Let the ordinal pattern probabilities be $q_1,\ldots,q_6$.

Permutation entropy is:

$$
H_c = -\sum_{j=1}^{6}q_j\log(q_j)
$$

using natural logarithms.

Normalized permutation entropy is:

$$
H_c^{norm} = \frac{H_c}{\log(6)}
$$

so that:

$$
0 \leq H_c^{norm} \leq 1
$$

## Outputs

Entropy outputs are saved under `results/global_diagnosis/entropy`:

```text
results/global_diagnosis/entropy/layer_entropy.csv
results/global_diagnosis/entropy/entropy_report.json
```

The empirical entropy CSV has one row per component:

```text
series
component
k
component_type
scale_minutes
scale_days
repeat_length
effective_n
ordinal_windows
permutation_entropy
normalized_entropy
```

For detail components, `scale_minutes` follows the smaller-time convention:

$$
\text{scale\_minutes}(D_k) = 5 \times 2^{k-1}
$$

For the approximation component:

$$
\text{scale\_minutes}(A_{11}) = 5 \times 2^{11}
$$

Ordinal pattern counts are recorded in `results/global_diagnosis/entropy/entropy_report.json`.

## Results

Normalized entropy is high across all final EUR/USD components:

```text
final D_01 normalized_entropy: 0.9905144523650508
final D_06 normalized_entropy: 0.9894543018698768
final D_11 normalized_entropy: 0.9886595540022752
final A_11 normalized_entropy: 0.998564063534312
```

Effective sample size decreases with scale because repeated block values are
compressed before entropy calculation:

```text
D_01 effective_n: 735,232
D_06 effective_n: 22,976
D_11 effective_n: 718
A_11 effective_n: 359
```

Coarse-scale entropy estimates are therefore treated as noisier than fine-scale
entropy estimates.

---

# Monte Carlo Metric Summaries and Comparisons

Objective: summarize baseline variability and compare empirical EUR/USD
diagnostics against the simulated baseline distributions.

## Simulation-level metric tables

Every baseline simulation is run through the same metric calculations as the
empirical series. Simulation-level outputs are saved under
`results/global_diagnosis/monte_carlo_baselines`:

```text
layer_volatility_simulations.csv
layer_entropy_simulations.csv
acf_simulations.csv
component_acf_simulations.csv
abs_component_correlation_simulations.csv
```

The simulation tables include `baseline_type` and `simulation_id`, along with
the relevant component, lag, metric, or component-pair identifiers.

## Summary tables

For each baseline type and each pointwise diagnostic, V1.1 computes:

```text
mean
median
std
p05
p95
min
max
```

Quantiles use linear interpolation:

```python
np.quantile(values, [0.05, 0.5, 0.95], method="linear")
```

Summary outputs are:

```text
layer_volatility_summary.csv
layer_entropy_summary.csv
acf_summary.csv
component_acf_summary.csv
abs_component_correlation_summary.csv
```

Plots use the baseline median and the 5-95% envelope. The mean, standard
deviation, minimum, and maximum are retained for exploratory analysis and sanity
checks.

The 5-95% bands are Monte Carlo baseline envelopes, not confidence intervals for
the empirical EUR/USD statistic. They describe the distribution of each
diagnostic under the chosen baseline-generating process.

## Empirical comparison tables

For each empirical diagnostic value, V1.1 records:

```text
empirical_value
baseline_median
baseline_p05
baseline_p95
difference_from_median
percentile_rank
inside_envelope
above_envelope
below_envelope
outside_envelope
```

Percentile rank is:

$$
\frac{1}{100}\sum_{m=0}^{99}\mathbf{1}\{x_m^{baseline} \leq x^{EURUSD}\}
$$

Ties are counted with `<=`. Percentile rank is an empirical baseline diagnostic,
not a formal p-value.

Comparison outputs are:

```text
layer_volatility_empirical_comparison.csv
layer_entropy_empirical_comparison.csv
acf_empirical_comparison.csv
component_acf_empirical_comparison.csv
abs_component_correlation_empirical_comparison.csv
```

## Runtime logging

Full pipeline timing is recorded in:

```text
results/global_diagnosis/monte_carlo_baselines/runtime_log.csv
```

Monte Carlo metric timings are split into reading, volatility, entropy, return
ACF, component ACF, absolute component correlation, and simulation-total stages.
Entropy is currently the slowest Monte Carlo metric stage.

---

# Rolling Window Diagnostics

Objective: inspect time-local multi-scale volatility structure using fixed-size
rolling windows over the standardized return series $R^{\ast}$.

## Design choices

Rolling windows are defined on observation index, not calendar time.

For window length $W$ and step size $S$:

$$
\mathcal{W}_{q,W} = \{r_q^{\ast}, r_{q+1}^{\ast}, \ldots, r_{q+W-1}^{\ast}\}
$$

with:

```text
W in {2048, 8192}
S = 288
K_roll = 9
```

The current exact window counts are:

```text
W=2048: 2546 windows
W=8192: 2525 windows
```

Each rolling window is decomposed locally:

$$
\mathcal{W}_{q,W} = A_9^{(q,W)} + \sum_{k=1}^{9}D_k^{(q,W)}
$$

The reconstruction check requires:

$$
\max_i |w_i - \hat{w}_i| \leq 10^{-12}
$$

where:

$$
\hat{w} = A_9^{(q,W)} + \sum_{k=1}^{9}D_k^{(q,W)}
$$

## Rolling metrics

For each window and component, V2.1 computes RMS volatility:

$$
\sigma_{c,q,W}^{RMS} = \sqrt{\frac{1}{W}\sum_{i=1}^{W}x_{c,i}^{2}}
$$

for:

$$
c \in \{D_1,\ldots,D_9,A_9\}
$$

Detail energy share is computed over detail components only:

$$
p_{k,q,W}^{detail} =
\frac{E(D_k^{(q,W)})}{\sum_{j=1}^{9}E(D_j^{(q,W)})}
$$

Total component energy share includes the approximation component:

$$
p_{c,q,W}^{total} =
\frac{E(c^{(q,W)})}{\sum_{j=1}^{9}E(D_j^{(q,W)})+E(A_9^{(q,W)})}
$$

The detail scales are also summarized into three groups:

```text
fine:   D_1, D_2, D_3
mid:    D_4, D_5, D_6
coarse: D_7, D_8, D_9
```

For each window:

$$
p_{fine,q,W} + p_{mid,q,W} + p_{coarse,q,W} = 1
$$

up to floating-point tolerance.

## Outputs

Rolling outputs are saved under `results/rolling_window_diagnosis/rolling_metrics`:

```text
rolling_window_metadata.csv
rolling_layer_volatility.csv
rolling_window_summary.csv
rolling_scale_group_summary.csv
rolling_example_windows.csv
rolling_report.json
```

---

# Rolling Baseline Correlation Envelopes

Objective: compare empirical rolling cross-component correlation structure
against the existing shuffled and Gaussian Monte Carlo baselines.

For each baseline simulation, V2.1 reuses the baseline return parquet files and
applies the same rolling windows, local decomposition depth, and rolling metrics
as the empirical series.

For each baseline type, simulation, and window length, two correlation matrices
are computed.

The first uses rolling RMS volatility:

$$
\rho_{c,d,W}^{RMS,m} =
\mathrm{Corr}\left(\sigma_{c,\cdot,W}^{RMS,m},
\sigma_{d,\cdot,W}^{RMS,m}\right)
$$

The second uses percentile-ranked rolling detail energy shares. For each detail
component, percentile ranks are computed within the same window length:

$$
u_{k,q,W} =
\frac{1}{Q_W}\sum_{q'=1}^{Q_W}\mathbf{1}\{p_{k,q',W}^{detail} \leq p_{k,q,W}^{detail}\}
$$

Then:

$$
\rho_{k,l,W}^{share,pct,m} =
\mathrm{Corr}(u_{k,\cdot,W}^{m},u_{l,\cdot,W}^{m})
$$

For each baseline type and component pair, V2.1 records:

```text
median
p05
p95
inside_envelope
above_envelope
below_envelope
outside_envelope
```

These are Monte Carlo envelopes for rolling correlation structure, not formal
hypothesis tests.

Rolling baseline outputs are saved under `results/rolling_window_diagnosis/rolling_baselines`:

```text
rolling_correlation_simulations.csv
rolling_correlation_summary.csv
rolling_correlation_empirical_comparison.csv
runtime_log.csv
rolling_baseline_report.json
```

The runtime log records per-simulation stages so that window generation,
decomposition, metric calculation, and correlation calculation bottlenecks can
be inspected after a full run.

---

# Plot Reference

All return-series plots use observation index rather than timestamp on the
x-axis unless stated otherwise. Timestamps are retained in the underlying
datasets for later event lookup, but they are not used as plot axes in this
version.

## Return EDA Plots

Folder:

```text
plots/results/global_diagnosis/data_eda/returns
```

**Return line plots**

```text
final_returns_line.png
```

This plot shows the empirical standardized return series:

$$
R^{\ast}
$$

x-axis:

$$
i = 1,\ldots,N^{\ast}
$$

y-axis:

$$
r_i
$$

**Final vs Gaussian histogram**

```text
final_vs_gaussian_histogram.png
```

This plot compares the empirical distribution of:

$$
R^{\ast}
$$

against:

$$
R^{BM}
$$

using density-normalized histograms.

**Final vs Gaussian ECDF**

```text
final_vs_gaussian_ecdf.png
```

For a return series $R$, the empirical cumulative distribution function is:

$$
\hat{F}(x) = \frac{1}{N^{\ast}}\sum_{i=1}^{N^{\ast}}\mathbf{1}\{r_i \leq x\}
$$

The plot compares:

$$
\hat{F}_{EURUSD}(x)
$$

and:

$$
\hat{F}_{BM}(x)
$$

**Final QQ plot against Gaussian**

```text
final_qq_gaussian.png
```

The x-axis is the theoretical quantile from:

$$
\mathcal{N}(0, \mathrm{Var}(R^{\ast}))
$$

The y-axis is the corresponding empirical quantile of:

$$
R^{\ast}
$$

The diagonal reference line is:

$$
y=x
$$

**Return autocorrelation plots**

```text
final_vs_baselines_returns_acf.png
final_vs_baselines_abs_returns_acf.png
```

For a series $X_i$, autocorrelation at lag $\ell$ is:

$$
\rho_X(\ell) = \mathrm{Corr}(X_i, X_{i-\ell})
$$

The first plot uses:

$$
X_i = r_i
$$

The second plot uses:

$$
X_i = |r_i|
$$

In V1.1, each plot compares the empirical ACF line against shuffled and Gaussian
Monte Carlo baseline medians with 5-95% envelopes.

## Decomposition EDA Plots

Folder:

```text
plots/results/global_diagnosis/data_eda/decomposition
```

Let:

$$
X_c \in \{D_1,\ldots,D_{11},A_{11}\}
$$

denote a decomposition component.

**Layer plots**

```text
final_layers.png
shuffle_layers.png
gaussian_layers.png
```

Each figure contains stacked panels for:

```text
original
D_01
...
D_11
A_11
```

x-axis:

$$
i = 1,\ldots,N^{\ast}
$$

y-axis for component $c$:

$$
x_{c,i}
$$

**Layer distribution grid**

```text
layer_histograms_grid.png
```

This is a $3 \times 4$ grid over:

$$
D_1,\ldots,D_{11},A_{11}
$$

Each subplot compares the density-normalized distribution of the final EUR/USD
component against the Gaussian baseline component.

**Layer QQ grid**

```text
layer_qq_gaussian_grid.png
```

This is a $3 \times 4$ grid over:

$$
D_1,\ldots,D_{11},A_{11}
$$

For each component $c$, the x-axis is the theoretical quantile from:

$$
\mathcal{N}(0, \mathrm{Var}(X_c^{EURUSD}))
$$

and the y-axis is the empirical quantile of:

$$
X_c^{EURUSD}
$$

**Layer autocorrelation grids**

```text
layer_acf_returns_short_scales.png
layer_acf_abs_returns_short_scales.png
layer_acf_returns_long_scales.png
layer_acf_abs_returns_long_scales.png
```

For a component $X_c$, the signed-component autocorrelation is:

$$
\rho_c(\ell) = \mathrm{Corr}(x_{c,i}, x_{c,i-\ell})
$$

The absolute-component autocorrelation is:

$$
\rho_c^{abs}(\ell) = \mathrm{Corr}(|x_{c,i}|, |x_{c,i-\ell}|)
$$

Short-scale ACF plots contain:

$$
D_1,\ldots,D_6
$$

Long-scale ACF plots contain:

$$
D_7,\ldots,D_{11},A_{11}
$$

For larger-scale components, deterministic repeated block values are compressed
before computing autocorrelation. The x-axis is then mapped back to original
5-minute index lags.

In V1.1, these grids show empirical component ACF lines against shuffled and
Gaussian Monte Carlo baseline medians with 5-95% envelopes.

**Absolute component correlation heatmaps**

Folder:

```text
plots/results/global_diagnosis/correlation
```

```text
abs_corr_empirical.png
abs_corr_empirical_minus_shuffle_median.png
abs_corr_outside_shuffle_envelope.png
abs_corr_empirical_minus_gaussian_median.png
abs_corr_outside_gaussian_envelope.png
```

The empirical matrix is:

$$
\rho_{c,d}^{abs} = \mathrm{Corr}(|X_c|, |X_d|)
$$

where:

$$
c,d \in \{D_1,\ldots,D_{11},A_{11}\}
$$

The difference matrices are:

$$
\rho_{c,d}^{EURUSD,abs} - \mathrm{median}_m(\rho_{c,d}^{baseline,m,abs})
$$

Positive values mean the final EUR/USD components have higher absolute
cross-component correlation than the baseline median for that component pair.
Outside-envelope heatmaps mark whether the empirical value lies outside the
baseline 5-95% envelope.

## Volatility Result Plots

Folder:

```text
plots/results/global_diagnosis/volatility
```

All volatility plots use categorical component x-axis.

**Energy-share plots**

```text
detail_energy_share.png
total_component_energy_share.png
detail_energy_share_difference.png
total_component_energy_share_difference.png
```

`detail_energy_share.png` plots:

$$
p_k^{detail} = \frac{E(D_k)}{\sum_{j=1}^{11}E(D_j)}
$$

for:

$$
D_1,\ldots,D_{11}
$$

`total_component_energy_share.png` plots:

$$
p_c^{total} = \frac{E_c}{\sum_{j=1}^{11}E(D_j)+E(A_{11})}
$$

for:

$$
c \in \{D_1,\ldots,D_{11},A_{11}\}
$$

In V1.1, the level plots show empirical EUR/USD with shuffled and Gaussian
baseline medians and 5-95% envelopes.

The difference plots show empirical EUR/USD minus the baseline median:

$$
p_c^{EURUSD} - \mathrm{median}_m(p_c^{shuffle,m})
$$

and:

$$
p_c^{EURUSD} - \mathrm{median}_m(p_c^{BM,m})
$$

with a horizontal zero reference line.

**RMS volatility plots**

```text
rms_volatility.png
annualized_rms_volatility.png
rms_volatility_difference.png
```

`rms_volatility.png` plots:

$$
\sigma_c^{RMS} = \sqrt{\frac{1}{N^{\ast}}E_c}
$$

for:

$$
c \in \{D_1,\ldots,D_{11},A_{11}\}
$$

`annualized_rms_volatility.png` plots:

$$
\sigma_{c,ann}^{RMS} = \sigma_c^{RMS}\sqrt{252 \times 24 \times 12}
$$

In V1.1, `rms_volatility.png` and `annualized_rms_volatility.png` show baseline
medians and 5-95% envelopes. `rms_volatility_difference.png` plots:

$$
\sigma_c^{EURUSD,RMS} - \mathrm{median}_m(\sigma_c^{shuffle,m,RMS})
$$

and:

$$
\sigma_c^{EURUSD,RMS} - \mathrm{median}_m(\sigma_c^{BM,m,RMS})
$$

## Entropy Result Plots

Folder:

```text
plots/results/global_diagnosis/entropy
```

All entropy plots use categorical component x-axis.

**Layer entropy plots**

```text
permutation_entropy.png
normalized_entropy.png
```

`permutation_entropy.png` plots:

$$
H_c = -\sum_{j=1}^{6}q_j\log(q_j)
$$

for:

$$
c \in \{D_1,\ldots,D_{11},A_{11}\}
$$

`normalized_entropy.png` plots:

$$
H_c^{norm} = \frac{H_c}{\log(6)}
$$

In V1.1, both plots compare the empirical EUR/USD entropy profile against
shuffled and Gaussian Monte Carlo baseline medians with 5-95% envelopes.

**Entropy gap plot**

```text
entropy_gaps.png
```

In V1.1, this plot shows empirical EUR/USD minus the baseline median:

$$
\Delta H_c^{shuffle} = H_c^{EURUSD,norm} - \mathrm{median}_m(H_c^{shuffle,m,norm})
$$

and:

$$
\Delta H_c^{BM} = H_c^{EURUSD,norm} - \mathrm{median}_m(H_c^{BM,m,norm})
$$

The horizontal reference line is:

$$
\Delta H = 0
$$

**Ordinal pattern distribution grids**

```text
final_pattern_distribution.png
```

The empirical pattern grid is a $3 \times 4$ grid over:

$$
D_1,\ldots,D_{11},A_{11}
$$

For each component, the bar heights are ordinal-pattern shares:

$$
\hat{q}_j = \frac{n_j}{n_{\mathrm{windows}}}
$$

for the six possible patterns:

```text
012
021
102
120
201
210
```

The dashed reference line is the uniform share:

$$
\frac{1}{6}
$$

## Rolling Window Plots

Folder:

```text
plots/results/rolling_window_diagnosis
```

Rolling plots use rolling window index on the x-axis. Window metadata in
`results/rolling_window_diagnosis/rolling_metrics/rolling_window_metadata.csv` can be used to map a window back
to its start and end timestamps.

**Total volatility plots**

```text
rolling_total_volatility_2048.png
rolling_total_volatility_8192.png
window_length_total_volatility_comparison.png
```

These plots show the rolling RMS volatility of the original window:

$$
\sigma_{q,W}^{RMS} = \sqrt{\frac{1}{W}\sum_{i=1}^{W}w_{q,i}^{2}}
$$

**Rolling RMS plots**

Folder:

```text
plots/results/rolling_window_diagnosis/rms
```

```text
rms_volatility_heatmap_2048.png
rms_volatility_heatmap_8192.png
rms_volatility_percentile_heatmap_2048.png
rms_volatility_percentile_heatmap_8192.png
rms_volatility_zscore_heatmap_2048.png
rms_volatility_zscore_heatmap_8192.png
rms_volatility_correlation_2048.png
rms_volatility_correlation_8192.png
rms_volatility_percentile_correlation_2048.png
rms_volatility_percentile_correlation_8192.png
rms_volatility_zscore_correlation_2048.png
rms_volatility_zscore_correlation_8192.png
selected_scale_rms_2048.png
selected_scale_rms_8192.png
```

Percentile and z-score heatmaps are computed within each window length and
component. They are EDA views of relative rolling intensity, not replacements
for the raw RMS heatmaps.

**Rolling energy-share plots**

Folder:

```text
plots/results/rolling_window_diagnosis/energy_share
```

```text
detail_energy_share_heatmap_2048.png
detail_energy_share_heatmap_8192.png
detail_energy_share_percentile_heatmap_2048.png
detail_energy_share_percentile_heatmap_8192.png
detail_energy_share_percentile_correlation_2048.png
detail_energy_share_percentile_correlation_8192.png
fine_mid_coarse_share_2048.png
fine_mid_coarse_share_8192.png
fine_mid_coarse_share_correlation_2048.png
fine_mid_coarse_share_correlation_8192.png
window_length_fine_share_comparison.png
window_length_mid_share_comparison.png
window_length_coarse_share_comparison.png
```

The fine/mid/coarse correlation heatmaps include total window RMS alongside the
three group shares:

$$
\{\sigma_{q,W}^{RMS}, p_{fine,q,W}, p_{mid,q,W}, p_{coarse,q,W}\}
$$

**Rolling example decompositions**

Folder:

```text
plots/results/rolling_window_diagnosis/examples
```

Example plots show all rolling decomposition scales for selected windows,
including first, last, maximum-volatility, minimum-volatility, median-volatility,
and random EDA windows.

**Rolling baseline comparison plots**

Folder:

```text
plots/results/rolling_window_diagnosis/rolling_baselines
```

The `rms` and `energy_share` subfolders contain empirical rolling correlation
matrices, shuffled and Gaussian baseline medians, empirical-minus-median
matrices, and outside-envelope indicators for the rolling baseline comparisons.

## Memo Plots

Folder:

```text
plots/memo
```

Memo plots are presentation-oriented figures used in `Memo.md`. They are derived
from the same datasets and result tables documented above.

**Figure 1: Decomposition example**

```text
figure_01_decomposition_example.png
```

Shows selected components from the EUR/USD decomposition:

$$
R^{\ast},\quad D_1,\quad D_3,\quad D_6,\quad D_9,\quad D_{11},\quad A_{11}
$$

with observation index on the x-axis.

**Figure 2: Return distribution**

```text
figure_02_return_distribution.png
```

Combines a QQ plot of $R^{\ast}$ against:

$$
\mathcal{N}(0, \mathrm{Var}(R^{\ast}))
$$

with a zoomed density histogram comparing:

$$
R^{\ast}
$$

against:

$$
R^{BM}
$$

**Figure 3: Absolute-return autocorrelation**

```text
figure_03_abs_return_acf.png
```

Plots empirical absolute-return autocorrelation against shuffled and Gaussian
Monte Carlo baseline medians with 5-95% envelopes:

$$
\mathrm{Corr}(|r_i|, |r_{i-\ell}|)
$$

**Figure 4: Energy profile**

```text
figure_04_energy_profile.png
```

Shows empirical detail energy share with shuffled and Gaussian baseline medians
and 5-95% envelopes:

$$
p_k^{detail} = \frac{E(D_k)}{\sum_{j=1}^{11}E(D_j)}
$$

The second panel shows empirical detail energy share minus each baseline median.
The shaded bands are transformed into the same excess scale:

$$
p_k^{EURUSD} - p_{k,95}^{baseline} \quad \text{to} \quad p_k^{EURUSD} - p_{k,05}^{baseline}
$$

**Figure 5: Cross-scale correlation**

```text
figure_05_cross_scale_correlation.png
```

Shows:

$$
\mathrm{Corr}(|X_c|, |X_d|)
$$

for EUR/USD components, the difference against the shuffled baseline median, and
an indicator for component pairs outside the shuffled 5-95% envelope.

**Figure 6: Entropy profile**

```text
figure_06_entropy_profile.png
```

Shows normalized permutation entropy:

$$
H_c^{norm} = \frac{H_c}{\log(6)}
$$

for EUR/USD and shuffled/Gaussian baseline medians with 5-95% envelopes. The
dashed reference line uses:

$$
q = \left(\frac{1}{8}, \frac{3}{16}, \frac{3}{16}, \frac{3}{16}, \frac{3}{16}, \frac{1}{8}\right)
$$

and:

$$
H_{ref}^{norm} = \frac{-\sum_{j=1}^{6}q_j\log(q_j)}{\log(6)} \approx 0.9908
$$
