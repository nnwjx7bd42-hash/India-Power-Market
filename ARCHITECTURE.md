# Architecture: BESS Arbitrage Optimization Stack

This document describes the technical architecture of each layer, the design rationale, and how the layers connect.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Layer 1: Data Pipeline](#layer-1-data-pipeline)
3. [Layer 2: Point Forecasting](#layer-2-point-forecasting)
4. [Layer 3: Probabilistic Forecasting Engine](#layer-3-probabilistic-forecasting-engine)
5. [Layer 4: Stochastic BESS Optimizer](#layer-4-stochastic-bess-optimizer)
6. [Layer 5: Realistic Merchant Optimizer](#layer-5-realistic-merchant-optimizer)
7. [Backtest Methodology](#backtest-methodology)

---

## System Overview

The stack solves a single problem: **given a week of uncertain future electricity prices, how should a battery storage system bid into the IEX Day-Ahead Market to maximise risk-adjusted revenue?**

The solution is decomposed into five layers, each building on the previous:

```
Data → Point Forecasts → Probabilistic Forecasts → Price Scenarios → Optimal Dispatch
```

Each layer is independently testable and versioned (`v2/`, `v4/`, `v5/`, `v6/`).

---

## Layer 1: Data Pipeline

**Location**: `src/data_pipeline/`, `src/utils/`

### Data Sources

| Source | Granularity | Period | Features |
|--------|-------------|--------|----------|
| IEX DAM prices | Hourly | Sep 2021 – Jun 2025 | Clearing price (INR/MWh) |
| NERLDC/POSOCO | Hourly | Sep 2021 – Jun 2025 | Demand, supply, RE generation |
| Open-Meteo API | Hourly | Sep 2021 – Jun 2025 | Temperature, radiation, wind (5 cities) |
| Calendar | Hourly | — | Hour, day-of-week, holidays, season |

### Feature Engineering

The pipeline produces ~25 features per hour:

- **Calendar**: hour-of-day (sine/cosine encoded), day-of-week, weekend flag, Indian holiday flag, season, month
- **Weather**: national-average temperature, cooling degree hours (CDH), direct solar radiation, wind speed
- **Load**: system demand, net load (demand minus RE), RE penetration ratio, solar ramp rate
- **Price anchors** (long-memory): same-hour-last-week price, 2-week-ago price, rolling 4-week mean/median/std/P90 by hour-of-day
- **Interactions**: Hour x CDH, RE penetration x Hour

### Key Design Choice

Near-term lags (`P(T-1)`, `P(T-2)`, `P(T-24)`) are **deliberately excluded** from the planning feature set. At DAM bid submission time (10:00–12:00, day D-1 for delivery on day D), you don't have yesterday's clearing prices yet. The v4 probabilistic engine relies solely on exogenous features and long-memory anchors.

---

## Layer 2: Point Forecasting

**Location**: `v2/` (point models), `v3/` (ensemble)

Two point-forecast models, tuned independently:

### LSTM (PyTorch)

- Sequence-to-one architecture with configurable hidden layers, dropout
- Bayesian hyperparameter search via Optuna (~100 trials)
- Nested temporal cross-validation (3 expanding folds) to avoid selection bias
- Multi-seed stability testing (5 seeds) to measure variance

### XGBoost

- Gradient boosted trees with temporal train/val/test split
- Tuned via Optuna with early stopping

### V3 Ensemble

- **Location**: `v3/` at repo root (run_ensemble.py, ensemble_predict.py, ensemble_analysis.py)
- Weighted average of LSTM and XGBoost predictions
- Weights optimised on validation set (minimise MAPE)

### Evaluation Metrics

- MAPE, MAE, RMSE, R-squared
- Mean Directional Accuracy (MDA)
- Tolerance-band accuracy (within +/-5%, +/-10%)
- Forecast bias decomposition (over-forecasting vs under-forecasting)

---

## Layer 3: Probabilistic Forecasting Engine

**Location**: `v4/`

The point forecasts from Layer 2 are useful for diagnostics but insufficient for optimisation under uncertainty. Layer 3 produces a **full conditional distribution** of week-ahead prices.

### Step 1: Multi-Quantile XGBoost

- Single XGBoost model with `objective="reg:quantileerror"` and 11 quantile levels: `[0.05, 0.10, 0.20, ..., 0.90, 0.95]`
- Isotonic rearrangement (`np.sort`) post-prediction to prevent quantile crossing
- Evaluated with: pinball loss, CRPS, empirical coverage, Winkler score, PIT histogram

### Step 2: Conformal Calibration

- **Adaptive Conformal Inference (ACI)**: wraps the raw quantile forecasts and adjusts them online to achieve guaranteed asymptotic coverage
- Uses the Conformal PID variant with a learning rate parameter
- Calibrated on the last N weeks of validation data
- Produces a calibration plot (reliability diagram) and PIT histogram

### Step 3: Gaussian Copula Scenario Generation

The optimiser needs discrete price *paths* (scenarios), not marginal distributions at each hour.

1. **Historical residuals**: compute quantile residuals (PIT values) from the training/validation period
2. **Rank correlation matrix**: estimate Spearman correlation across the 168 hours of a week from rolling historical residuals
3. **Correlated uniform samples**: draw from a Gaussian copula using Cholesky decomposition, then map through the normal CDF to get correlated uniforms
4. **Map to prices**: interpolate each hour's uniform through the calibrated quantile function to get a price

This produces **200 raw scenarios** (168-hour price paths).

### Step 4: Scenario Reduction

Forward reduction algorithm selects a representative subset:

- Start with the full set of 200 scenarios
- Iteratively remove the scenario that is "closest" (Euclidean distance) to remaining scenarios, redistributing its probability weight
- Continue until **10 representative scenarios** remain
- Each scenario carries a probability weight proportional to the mass it absorbed

---

## Layer 4: Stochastic BESS Optimizer

**Location**: `v5/`

### Problem Formulation

Given 10 weighted price scenarios over a 168-hour horizon, find the charge/discharge schedule that maximises risk-adjusted revenue for a battery system.

### Decision Variables

- `p_ch[t]`: charge power at hour t (MW), `0 <= p_ch[t] <= P_max`
- `p_dis[t]`: discharge power at hour t (MW), `0 <= p_dis[t] <= P_max`
- `soc[t]`: state of charge at hour t (MWh)
- `zeta`: CVaR auxiliary variable (Value-at-Risk threshold)
- `u[s]`: per-scenario shortfall variables

### Objective

```
max  (1-beta) * E[Revenue] + beta * CVaR_alpha[Revenue]
```

where:

- `E[Revenue] = sum_s w_s * R_s` (probability-weighted expected revenue)
- `CVaR_alpha = zeta - (1/(1-alpha)) * sum_s w_s * u_s` (Rockafellar-Uryasev linearisation)
- `beta` controls risk aversion (0 = risk-neutral, 1 = fully risk-averse)

### Constraints

- **SoC dynamics**: `soc[t] = soc[t-1] + eta_ch * p_ch[t] - p_dis[t] / eta_dis`
- **SoC bounds**: `E_min <= soc[t] <= E_max`
- **Terminal SoC**: `soc[T] >= soc[0] - tolerance` (prevent end-of-horizon dumping)
- **CVaR auxiliary**: `u[s] >= zeta - R_s` for all scenarios s

### Why Pure LP (No Binaries)?

IEX DAM prices are non-negative. This means a rational agent never charges and discharges simultaneously — the LP relaxation is tight, and no binary mutual-exclusion constraint is needed. This makes the solve instantaneous (~10ms per week).

### Baselines

- **Perfect foresight**: deterministic LP with actual realised prices (upper bound)
- **Naive threshold**: charge below INR 3000/MWh, discharge above INR 6000/MWh
- **Risk-neutral**: same stochastic LP with `beta=0`
- **Columbia consensus**: Monte Carlo voting heuristic (sample 100 price paths from conformal intervals, solve 100 deterministic LPs, take majority-vote action)

---

## Layer 5: Realistic Merchant Optimizer

**Location**: `v6/`

Layer 5 takes the stochastic optimiser and adds all the real-world costs and constraints that determine actual merchant BESS P&L in India.

### BESS Physical Parameters

- 20 MW / 40 MWh LFP system (2C rate)
- Usable capacity: 32 MWh (10–90% SoC window)
- Charge/discharge efficiency: 92.2% one-way (RTE ~85% Year 1)
- RTE degradation: 0.25%/year; capacity fade: 2%/year
- Throughput degradation cost: INR 1,471/MWh

### Cycle Limits (Hard Constraints)

- 2 cycles/day maximum
- 8.08 cycles/week (= 420/year / 52 weeks)
- Modelled as hard constraints in the LP, not soft penalties

### IEX DAM Bidding

- 15-minute time blocks (96 per day)
- Volume granularity: 0.1 MW steps
- Price floor: INR 0/MWh, ceiling: INR 10,000/MWh
- Hourly LP decisions are discretised to 15-minute IEX bid format

### Transaction Costs

| Cost Component | Value |
|----------------|-------|
| IEX transaction fee | INR 20/MWh (on all energy traded) |
| SLDC charges | INR 5/MWh |
| RLDC charges | INR 2/MWh |
| Transmission losses | 3% of delivered energy |

### CERC Deviation Settlement Mechanism (DSM)

The optimizer uses a conservative flat buffer (INR 25/MWh) during optimisation. The backtest simulator applies the full CERC slab structure:

| Deviation Band | Under-injection Penalty | Over-injection Penalty |
|----------------|------------------------|----------------------|
| 0–12% | Normal rate | Normal rate |
| 12–25% | 110% of normal rate | 90% of normal rate |
| > 25% | Penalty rate | Floor rate |

Deviation is simulated stochastically in the backtest (drawn from a truncated normal distribution around the schedule).

### Revenue Decomposition

Every week in the backtest produces:

```
Gross Revenue  = sum(P_dis * price) - sum(P_ch * price)
 - IEX fees
 - SLDC/RLDC fees
 - Transmission losses
 - DSM costs
 - Degradation cost
 = Net Revenue
```

### Sensitivity Analyses

Two built-in sensitivity configs:

1. **Zero transmission loss** (`transaction_costs_tx0.yaml`): isolates the "pure strategy" revenue from regulatory drag
2. **Reduced cycle cap** (`bess_config_cycles7.yaml`): models tighter OEM warranty terms (7 cycles/week vs 8.08)

---

## Backtest Methodology

### Rolling Weekly Evaluation

- **Period**: Jan 2024 – Jun 2025 (75 weeks)
- **Horizon**: 168 hours (1 week) per optimisation
- **Re-estimation**: conformal calibrator and copula correlation matrix are re-estimated each week using the most recent available data
- **No model re-training**: the quantile XGBoost model is trained once on data through Dec 2024 and held fixed

### Load Forecasting in Backtest (No Look-Ahead)

At DAM bid-submission time, future hourly demand is **unknown**. The backtest simulates realistic week-ahead load inputs by replacing actual future load features (`Demand`, `Net_Load`, `RE_Penetration`, `Solar_Ramp`) with historical proxies:

1. **Primary proxy**: same-hour-last-week value (i.e., the value observed exactly 168 hours before)
2. **Fallback**: if last-week data is missing, use the mean of the same hour-of-day over the preceding 4 weeks

This ensures the quantile model's feature matrix contains only information that would genuinely be available at planning time, eliminating look-ahead bias.

### Performance Notes

- **Vectorised data pipeline**: missing generation estimation (`estimate_missing_generation`) uses a vectorised map approach, and weather aggregation (`aggregate_to_national`) uses `groupby` — both replacing row-by-row loops for 10-100x speedups on large datasets.

### Capture Ratio Definitions

All capture ratios use **simulated** revenues under the **same actual price path**:

- `capture_vs_pf_pct` = 100 * (strategy net revenue / perfect foresight net revenue)
- `capture_vs_naive_pct` = 100 * (strategy net revenue / naive baseline net revenue)

### Reported Metrics

Per-week: gross revenue, net revenue, all cost components, cycles, SoC statistics, capture ratios

Aggregate: annual net revenue, mean/std capture ratio, realised CVaR, max drawdown, negative weeks, cycles/year

---

## Testing

The `tests/` directory contains pytest smoke tests covering each major layer:

- **`test_data_pipeline.py`**: vectorised estimation and aggregation correctness
- **`test_optimizer.py`**: LP feasibility, SoC bounds, revenue positivity, cycle constraint enforcement for both deterministic and stochastic formulations
- **`test_backtest.py`**: look-ahead bias guard (verifies forecasted demand != actual future demand), metrics sanity checks

Run with: `python -m pytest tests/ -v`

---

## Known Limitations

1. **No intraday re-optimisation**: bids are submitted once for the entire next day. No RTM participation.
2. **No expanding-window re-training**: the quantile model is fixed throughout the backtest. Distribution shift (e.g., new solar capacity) is only partially captured by conformal recalibration.
3. **Simplified degradation**: linear throughput cost vs rainflow cycle counting.
4. **No ancillary services**: BESS revenue from frequency regulation or spinning reserves is not modelled.
5. **Single market**: IEX DAM only — no bilateral contracts, no green term-ahead market.
6. **Weather forecasts as actuals**: the backtest uses realised weather as a proxy for weather forecasts. Forecast error would widen prediction intervals.
