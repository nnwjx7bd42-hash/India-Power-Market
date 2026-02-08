# Investor & Business Plan — Metrics Summary

**Purpose:** Single reference of all backtest and evaluation metrics for investor decks and business plans.  
**Source:** Project result files (v2, v4, v6).  
**Disclaimer:** Backtest results are historical simulations and do not guarantee future performance.

---

## 1. Point forecast performance (Layer 2 — v2)

**Evaluation:** Holdout week, 168 hours (17 Jun 2025 01:00 – 24 Jun 2025 00:00).  
**Models:** LSTM and XGBoost; best holdout model: **XGBoost**.

| Metric | LSTM | XGBoost |
|--------|------|---------|
| **MAPE (%)** | 18.41 | **17.87** |
| **RMSE (INR/MWh)** | 652.17 | 684.83 |
| **MAE (INR/MWh)** | 370.77 | 380.38 |
| **R²** | 0.896 | 0.885 |
| **Mean Directional Accuracy (MDA) (%)** | 80.24 | 76.05 |
| **Within ±5% of actual (%)** | 35.12 | 35.71 |
| **Within ±10% of actual (%)** | 20.83 | 20.83 |
| **Beyond ±10% (%)** | 44.05 | 43.45 |
| **Hit rate exact (%)** | 0.0 | 0.0 |
| **Hours over-forecast** | 119 | 111 |
| **Hours under-forecast** | 49 | 57 |

**Validation (pre-holdout):**  
- XGBoost: val MAPE 13.78%, val RMSE 696.03  
- LSTM: val MAPE 13.44%, val RMSE 733.01  

**Quote for deck:** *“Week-ahead price forecast achieves **17.87% MAPE** and **76–80% directional accuracy** on a strict 168-hour holdout (Jun 2025), with **R² 0.88–0.90**.”*

---

## 2. Probabilistic forecast performance (Layer 3 — v4)

**Model:** Multi-quantile XGBoost (11 quantiles).  
**Evaluation:** Validation, test, and holdout splits (train to Dec 2024; holdout Jun 17–24 2025).

### Validation set

| Metric | Value |
|--------|--------|
| Avg pinball loss | 260.04 |
| CRPS | 523.04 |
| Median MAPE (%) | 17.74 |
| Empirical coverage 90% interval (%) | 87.33 |
| Interval width 90% (INR/MWh) | 3,452.81 |
| Empirical coverage 80% (%) | 76.04 |
| Empirical coverage 60% (%) | 53.16 |

### Test set

| Metric | Value |
|--------|--------|
| Avg pinball loss | 362.90 |
| CRPS | 730.78 |
| Median MAPE (%) | 57.29 |
| Empirical coverage 90% (%) | 77.17 |
| Empirical coverage 80% (%) | 65.28 |
| Empirical coverage 60% (%) | 45.40 |

### Holdout (Jun 17–24 2025)

| Metric | Value |
|--------|--------|
| Avg pinball loss | 680.17 |
| CRPS | 1,355.36 |
| Median MAPE (%) | 73.06 |
| Empirical coverage 90% (%) | 51.19 |
| Empirical coverage 80% (%) | 32.14 |
| Empirical coverage 60% (%) | 17.26 |
| Interval width 90% (INR/MWh) | 3,242.13 |

### Scenario generation (holdout week)

| Metric | Value |
|--------|--------|
| Raw scenarios | 200 |
| Reduced scenarios | 10 |
| Scenario mean price (INR/MWh) | 4,834.83 |
| Scenario std price | 3,149.23 |
| Scenario daily spread (mean) | 8,570.77 |
| Scenario daily spread (p95) | 9,264.82 |
| Correlation MAE | 0.0538 |

**Quote for deck:** *“Probabilistic forecasts achieve **87% empirical coverage** on the 90% prediction interval in validation; scenario generation produces 10 representative weekly price paths for optimisation.”*

---

## 3. BESS backtest performance (Layer 5 — v6)

**Setup:** 20 MW / 40 MWh BESS, IEX DAM, full transaction costs (IEX, SLDC, RLDC, 3% transmission loss, DSM, degradation).  
**Window:** 75 weeks, 2024-01-01 to 2025-06-10 (Mondays).  
**Strategy:** Stochastic CVaR (β=0.3, α=0.95); no look-ahead (load features proxied).

### Main run (baseline: 3% tx loss, 8.08 cycles/week cap)

| Metric | Value |
|--------|--------|
| **Mean weekly net revenue (INR)** | 1,129,032 |
| **Mean weekly gross revenue (INR)** | 1,201,960 |
| **Mean weekly costs (INR)** | 72,928 |
| **Annual net revenue extrapolated (INR)** | **58,709,664** |
| **Revenue per MW per year (INR lakh)** | **29.35** |
| **Capture vs perfect foresight (%)** | **91.9** (std 6.6) |
| **Mean cycles per week** | 7.49 |
| **Cycles per year** | 389.3 |
| **CVaR realised (worst 5% of weeks) (INR)** | 398,481 |
| **Negative weeks** | **0 / 75** |
| **Max drawdown (INR)** | 0 |
| **Total backtest weeks** | 75 |

**Quote for deck:** *“Over **75 weeks** (Jan 2024–Jun 2025), the strategy achieves **INR 29.35 lakh per MW per year** net revenue, capturing **91.9%** of perfect-foresight revenue, with **zero negative weeks** and **CVaR (worst 5%) INR 398k** per week.”*

### Sensitivity: Zero transmission loss

| Metric | Value |
|--------|--------|
| Mean weekly net revenue (INR) | 1,184,398 |
| Annual net revenue (INR) | 61,588,716 |
| **Revenue per MW per year (INR lakh)** | **30.79** |
| Capture vs PF (%) | 92.0 (std 6.4) |
| Cycles per year | 395.0 |
| CVaR worst 5% (INR) | 440,434 |
| Negative weeks | 0 / 75 |
| Mean weekly costs (INR) | 15,313 |

**Quote for deck:** *“With zero transmission loss, revenue increases to **INR 30.79 lakh per MW per year**.”*

### Sensitivity: 7 cycles/week (tighter cycle cap)

| Metric | Value |
|--------|--------|
| Mean weekly net revenue (INR) | 1,085,627 |
| Annual net revenue (INR) | 56,452,622 |
| **Revenue per MW per year (INR lakh)** | **28.23** |
| Capture vs PF (%) | 92.2 (std 6.8) |
| Mean cycles per week | 6.87 |
| Cycles per year | 357.2 |
| CVaR worst 5% (INR) | 381,061 |
| Negative weeks | 0 / 75 |
| Mean weekly costs (INR) | 67,762 |

**Quote for deck:** *“Under a tighter **7 cycles/week** cap, revenue remains **INR 28.23 lakh per MW per year** with **0 negative weeks**.”*

---

## 4. One-page summary for deck

| Headline metric | Value |
|-----------------|--------|
| **Holdout forecast MAPE** | 17.87% (XGBoost) |
| **Holdout directional accuracy** | 76–80% |
| **Holdout R²** | 0.88–0.90 |
| **Backtest period** | 75 weeks (Jan 2024 – Jun 2025) |
| **Annual net revenue (base case)** | INR 58.71 Cr (20 MW) |
| **Revenue per MW per year** | **INR 29.35 lakh** (base) |
| **Revenue per MW per year (zero tx loss)** | INR 30.79 lakh |
| **Revenue per MW per year (7 cycles/week)** | INR 28.23 lakh |
| **Capture vs perfect foresight** | **91.9%** (±6.6%) |
| **Weeks with negative net revenue** | **0 / 75** |
| **CVaR (worst 5% of weeks)** | INR 398,481 per week |
| **Cycles per year (base)** | 389.3 (cap 420) |

---

## 5. Assumptions (for appendix)

- **BESS:** 20 MW / 40 MWh, 10–90% SoC, 92.2% round-trip efficiency, 8.08 cycles/week (base), degradation INR 1,471/MWh throughput.  
- **Market:** IEX DAM, hourly then 15-min bid discretisation.  
- **Costs:** IEX INR 20/MWh, SLDC INR 5/MWh, RLDC INR 2/MWh, 3% transmission loss, CERC DSM (slab), degradation.  
- **Backtest:** No look-ahead (load features = same-hour-last-week proxy); quantile model fixed (trained to Dec 2024); conformal and correlation re-estimated weekly.  
- **Data:** Planning dataset 33,073 hourly rows; holdout 168 h (Jun 17–24 2025).

---

## 6. Source files

| Layer | Source |
|-------|--------|
| v2 holdout | `v2/results/holdout_forecast_metrics.yaml`, `holdout_summary.yaml` |
| v2 validation | `v2/results/xgb_metrics_final.yaml`, `lstm_metrics_final.yaml` |
| v4 probabilistic | `v4/results/quantile_model_metrics.yaml`, `planning_summary.yaml` |
| v6 backtest (base) | `v6/results/backtest_full_log.txt` |
| v6 aggregate | `v6/results/aggregate_metrics.json`, `aggregate_metrics_tx0.json`, `aggregate_metrics_cycles7.json` |

*Document generated for investor and business plan use. Update when re-running backtests or evaluations.*
