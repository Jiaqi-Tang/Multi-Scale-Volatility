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
`data/intermediate` folder:

```text
data/intermediate/eurusd_1m_utc_clean.csv
data/intermediate/eurusd_5m_ohlc_utc_nonempty.csv
data/intermediate/eurusd_5m_log_returns_clean.csv
data/intermediate/preprocessing_report.json
```

The clean 5-minute return dataset contains `timestamp_utc, close, log_return, previous_timestamp_utc, previous_close, delta_minutes, n_m1`, where `n_m1` records the number of observed 1-minute bars used to construct the current 5-minute close bar.

Dropped returns are not exported as a separate dataset. They are recorded only for
debugging and audit purposes in `data/intermediate/preprocessing_report.json`.

The final analysis dataset will be produced after length standardization and saved
as:

```text
data/final/eurusd_5m_log_returns_final.csv
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

The standardized final dataset is saved as: `data/final/eurusd_5m_log_returns_final.csv`

The truncation report is saved as: `data/final/truncation_report.json`

For $R^{\ast}$:

```text
mean_log_return: 1.381445428226864e-07
variance_log_return: 8.257150232019612e-08
std_log_return: 0.00028735257493225306
min_log_return: -0.0097635255221106
max_log_return: 0.0126936410859073
median_log_return: 0.0
skewness_log_return: 0.15778024187331044
kurtosis_log_return: 44.2160995969573
```

---

# Baseline Series

Objective: Create baseline time series such that entropy can be interpreted against a baseline entropy.

## Design choices

Baseline series are generated from the standardized final return series $R^{\ast}$.

All baseline series have length $|R^{baseline}| = N^{\ast}$, and use the same timestamp index as $R^{\ast}$.

The timestamps are retained as alignment metadata; the baseline computations are conducted on ordered return index:

$$
i = 1,2,\ldots,N^{\ast}
$$

## Shuffled Baseline

The shuffled baseline is a random permutation of the standardized returns:

$$
R^{shuffle} = \pi(R^{\ast})
$$

where $\pi$ is a random permutation.

Random seed: $137$

Properties preserved:

- same empirical distribution as $R^{\ast}$
- same mean and variance as $R^{\ast}$
- same minimum and maximum as $R^{\ast}$

Property destroyed:

- temporal ordering

Output:

```text
data/baselines/eurusd_5m_log_returns_shuffle.csv
```

## Brownian / Gaussian Baseline

The Gaussian baseline is generated as:

$$
R^{BM}_i \sim \mathcal{N}(0, \sigma_R^2)
$$

where:

$$
\sigma_R^2 = \mathrm{Var}(R^{\ast})
$$

The variance is the population variance of the standardized final return series:

$$
\sigma_R^2 = \frac{1}{N^{\ast}}\sum_{i=1}^{N^{\ast}}(r_i - \bar{r})^2
$$

where $\sigma_R^2 = 8.257150232019612 \times 10^{-8}$

Random seed: $271$

Properties targeted:

- same population variance as $R^{\ast}$
- Gaussian independent increments
- zero mean

Output:

```text
data/baselines/eurusd_5m_log_returns_gaussian.csv
```

The baseline report is saved as:

```text
data/baselines/baselines_report.json
```

## Results

Shuffled baseline:

```text
rows: 735,232
mean_log_return: 1.381445428226864e-07
population_variance_log_return: 8.257150232019612e-08
population_std_log_return: 0.00028735257493225306
min_log_return: -0.0097635255221106
max_log_return: 0.0126936410859073
```

Gaussian baseline:

```text
rows: 735,232
target_mean_log_return: 0.0
target_population_variance_log_return: 8.257150232019612e-08
realized_mean_log_return: 1.1733939639875955e-07
realized_population_variance_log_return: 8.249276080303108e-08
realized_population_std_log_return: 0.0002872155302260501
min_log_return: -0.0015116429800731662
max_log_return: 0.0013819795205624716
```

---

# Block-Average Multi-Scale Decomposition

Objective: Decompose the final return series and baseline series into scale-indexed
detail layers and a final approximation layer.

## Design choices

The decomposition is applied to:

$$
R^{\ast},\quad R^{shuffle},\quad R^{BM}
$$

For each scale:

$$
k = 1,2,\ldots,K \quad \text{ with }\,K = 11
$$

the block size is:

$$
B_k = 2^k
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

The decomposition outputs are:

```text
data/decomposition/final_decomposition.csv
data/decomposition/shuffle_decomposition.csv
data/decomposition/gaussian_decomposition.csv
data/decomposition/decomposition_report.json
```

## Validation

For each decomposed series, reconstruction error is computed as:

$$
\epsilon_i = \mathrm{original}_i - \left(A_{11,i} + \sum_{k=1}^{11}D_{k,i}\right)
$$

The decomposition fails if:

$$
\max_i |\epsilon_i| > 10^{-12}
$$

All three decompositions reconstruct to machine precision.

```text
final:
  max_abs_reconstruction_error: 3.469446951953614e-18
  mean_abs_reconstruction_error: 2.2820538114180538e-20

shuffle:
  max_abs_reconstruction_error: 1.734723475976807e-18
  mean_abs_reconstruction_error: 2.348456095110165e-20

gaussian:
  max_abs_reconstruction_error: 4.336808689942018e-19
  mean_abs_reconstruction_error: 2.874568672696032e-20
```

---

# Volatility / Energy Metrics

Objective: Quantify how return variation is distributed across decomposition
components.

## Design choices

Volatility metrics are computed for:

$$
D_1,\ldots,D_{11},A_{11}
$$

for each of:

$$
R^{\ast},\quad R^{shuffle},\quad R^{BM}
$$

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

Volatility outputs are saved under `results/volatility`:

```text
results/volatility/layer_volatility.csv
results/volatility/volatility_report.json
```

The volatility CSV has one row per series and component:

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

## Results

For the final EUR/USD series, most detail-layer energy is concentrated at the
finest scales:

```text
D_01 detail_energy_share: 0.5128464888450802
D_02 detail_energy_share: 0.24912090865546974
D_03 detail_energy_share: 0.12177649049781977
```

The final approximation energy is small relative to total component energy:

```text
A_11 total_component_energy_share: 0.00038432560825245565
```

The component energy sum reconstructs the original return energy to numerical
precision:

```text
final energy_reconstruction_gap: 1.5050460877574778e-14
shuffle energy_reconstruction_gap: 1.7520707107365752e-14
gaussian energy_reconstruction_gap: 2.0122792321330962e-14
```

---

# Permutation Entropy

Objective: Quantify temporal ordering structure within each decomposition
component.

## Design choices

Permutation entropy is computed for:

$$
D_1,\ldots,D_{11},A_{11}
$$

for each of:

$$
R^{\ast},\quad R^{shuffle},\quad R^{BM}
$$

The embedding dimension and are:

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

Entropy outputs are saved under `results/entropy`:

```text
results/entropy/layer_entropy.csv
results/entropy/entropy_report.json
```

The entropy CSV has one row per series and component:

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

Ordinal pattern counts are recorded in `results/entropy/entropy_report.json`.

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

# Entropy Gap Metrics

Objective: Compare EUR/USD ordering structure against the shuffled and Gaussian
baselines.

## Design choices

Entropy gaps are computed using normalized entropy.

Against the shuffled baseline:

$$
\Delta H_c^{shuffle} = H_c^{shuffle,norm} - H_c^{EURUSD,norm}
$$

Against the Gaussian baseline:

$$
\Delta H_c^{BM} = H_c^{BM,norm} - H_c^{EURUSD,norm}
$$

Positive values indicate that the baseline has higher normalized entropy than
EUR/USD at component $c$. Under this interpretation, positive gaps suggest more
temporal ordering structure in EUR/USD than in the baseline at that component.

Negative values indicate that EUR/USD has higher normalized entropy than the
baseline at that component.

## Outputs

Entropy gap results are saved as:

```text
results/entropy/entropy_gaps.csv
```

The entropy gap CSV contains:

```text
component
k
component_type
scale_minutes
scale_days
repeat_length
final_normalized_entropy
shuffle_normalized_entropy
gaussian_normalized_entropy
entropy_gap_shuffle
entropy_gap_gaussian
```

## Results

Entropy gaps are small across most components. Examples:

```text
D_01 entropy_gap_shuffle: 0.0002695456277962416
D_01 entropy_gap_gaussian: 0.00020682881885636384

D_06 entropy_gap_shuffle: 0.0019306019646548878
D_06 entropy_gap_gaussian: 0.0012652754256139431

D_11 entropy_gap_shuffle: 0.003101733050344002
D_11 entropy_gap_gaussian: -0.0003802058801183339

A_11 entropy_gap_shuffle: -0.00459063275952265
A_11 entropy_gap_gaussian: -0.00011979411067764012
```

The small magnitude of the gaps is consistent with the high normalized entropy
observed across all series and components.

---

# Plot Reference

All return-series plots use observation index rather than timestamp on the
x-axis unless stated otherwise. Timestamps are retained in the underlying
datasets for later event lookup, but they are not used as plot axes in this
version.

## Return EDA Plots

Folder:

```text
plots/eda/returns
```

**Return line plots**

```text
final_returns_line.png
shuffle_returns_line.png
gaussian_returns_line.png
```

Each plot shows one return series:

$$
R^*,\quad R^{shuffle},\quad R^{BM}
$$

respectively.

x-axis:

$$
i = 1,\ldots,N^*
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
R^*
$$

against:

$$
R^{BM}
$$

using density-normalized histograms. Vertical lines mark each series mean and
median.

**Final vs Gaussian ECDF**

```text
final_vs_gaussian_ecdf.png
```

For a return series $R$, the empirical cumulative distribution function is:

$$
\hat{F}(x)
=
\frac{1}{N^*}\sum_{i=1}^{N^*}\mathbf{1}\{r_i \leq x\}
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
\mathcal{N}(0, Var(R^*))
$$

The y-axis is the corresponding empirical quantile of:

$$
R^*
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
\rho_X(\ell)
=
Corr(X_i, X_{i-\ell})
$$

The first plot uses:

$$
X_i = r_i
$$

The second plot uses:

$$
X_i = |r_i|
$$

Each plot compares:

$$
R^*,\quad R^{shuffle},\quad R^{BM}
$$

Dashed reference bands are:

$$
\pm \frac{1.96}{\sqrt{N^*}}
$$

## Decomposition EDA Plots

Folder:

```text
plots/eda/decomposition
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
i = 1,\ldots,N^*
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
\mathcal{N}(0, Var(X_c^{EURUSD}))
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
\rho_c(\ell)
=
Corr(x_{c,i}, x_{c,i-\ell})
$$

The absolute-component autocorrelation is:

$$
\rho_c^{abs}(\ell)
=
Corr(|x_{c,i}|, |x_{c,i-\ell}|)
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

**Absolute component correlation heatmaps**

```text
final_abs_component_correlation.png
shuffle_abs_component_correlation.png
gaussian_abs_component_correlation.png
```

For each series, the plotted matrix is:

$$
\rho_{c,d}^{abs}
=
Corr(|X_c|, |X_d|)
$$

where:

$$
c,d \in \{D_1,\ldots,D_{11},A_{11}\}
$$

The values are computed on the fully expanded, index-aligned decomposition
components.

**Final minus shuffled absolute component correlation**

```text
final_minus_shuffle_abs_component_correlation.png
```

The plotted matrix is:

$$
\rho_{c,d}^{EURUSD,abs}
-
\rho_{c,d}^{shuffle,abs}
$$

Positive values mean the final EUR/USD components have higher absolute
cross-component correlation than the shuffled baseline for that component pair.

## Volatility Result Plots

Folder:

```text
plots/results/volatility
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
p_k^{detail}
=
\frac{E(D_k)}{\sum_{j=1}^{11}E(D_j)}
$$

for:

$$
D_1,\ldots,D_{11}
$$

`total_component_energy_share.png` plots:

$$
p_c^{total}
=
\frac{E_c}{\sum_{j=1}^{11}E(D_j)+E(A_{11})}
$$

for:

$$
c \in \{D_1,\ldots,D_{11},A_{11}\}
$$

The difference plots show:

$$
p_c^{EURUSD} - p_c^{shuffle}
$$

and:

$$
p_c^{EURUSD} - p_c^{BM}
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
\sigma_c^{RMS}
=
\sqrt{\frac{1}{N^*}E_c}
$$

for:

$$
c \in \{D_1,\ldots,D_{11},A_{11}\}
$$

`annualized_rms_volatility.png` plots:

$$
\sigma_{c,ann}^{RMS}
=
\sigma_c^{RMS}\sqrt{252 \times 24 \times 12}
$$

`rms_volatility_difference.png` plots:

$$
\sigma_c^{EURUSD,RMS} - \sigma_c^{shuffle,RMS}
$$

and:

$$
\sigma_c^{EURUSD,RMS} - \sigma_c^{BM,RMS}
$$

## Entropy Result Plots

Folder:

```text
plots/results/entropy
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
H_c^{norm}
=
\frac{H_c}{\log(6)}
$$

Both plots compare:

$$
R^*,\quad R^{shuffle},\quad R^{BM}
$$

**Entropy gap plot**

```text
entropy_gaps.png
```

This plot shows:

$$
\Delta H_c^{shuffle}
=
H_c^{shuffle,norm} - H_c^{EURUSD,norm}
$$

and:

$$
\Delta H_c^{BM}
=
H_c^{BM,norm} - H_c^{EURUSD,norm}
$$

The horizontal reference line is:

$$
\Delta H = 0
$$

**Ordinal pattern distribution grids**

```text
final_pattern_distribution.png
shuffle_pattern_distribution.png
gaussian_pattern_distribution.png
```

Each figure is a $3 \times 4$ grid over:

$$
D_1,\ldots,D_{11},A_{11}
$$

For each component, the bar heights are ordinal-pattern shares:

$$
\hat{q}_j
=
\frac{\#\{\text{ordinal windows with pattern }j\}}
{\#\{\text{ordinal windows}\}}
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
