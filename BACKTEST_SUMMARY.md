# Backtest Performance Summary: GENCO BESS VPP

This document provides the definitive realized performance analysis for the GENCO 50MW / 200MWh BESS optimization build across 143 days (Feb 1, 2025 – June 24, 2025).

---

## 💎 Executive Summary (Recalibrated System)
All results are based on **actual market actuals** (realized prices), where the optimizer's commitments were fixed and then evaluated against the real market.

- **Total Realized Net Revenue**: **₹198,073,460**
- **Average Daily Revenue**: ₹1,385,129 (₹1.38M)
- **Peak Day Revenue**: ₹4,187,981
- **Worst Case Day**: **+₹50,978** (Baseline naturally avoids losses)
- **Annualized Sharpe Ratio**: **37.39**
- **Average Daily Cycling**: 1.20 Cycles

---

## 📈 Efficient Frontier: Risk-Return Mapping
The following table shows how varying the risk-aversion coefficient ($\lambda$) affects actual realized performance.

| Risk Aversion ($\lambda$) | Net Revenue (₹M) | Worst Day (₹K) | Sharpe Ratio | Avg Daily Cycles |
| :--- | :--- | :--- | :--- | :--- |
| **0.00 (Baseline)** | **198.07** | **+50.98** | 37.39 | 1.20 |
| **0.05 (Cautious)** | 196.68 | +59.23 | 37.86 | 1.20 |
| **0.10 (Balanced)** | 195.26 | +63.63 | **37.97** | 1.20 |
| **0.30 (Risk-Averse)** | 190.50 | +77.62 | 37.41 | 1.20 |
| **0.50 (Defensive)** | 187.09 | **+138.72** | 37.52 | 1.20 |

---

## 📊 Statistical Distributions
Analysis of daily returns across the 143-day backtest period:

- **Percentile 10 (p10)**: ~₹563K / day
- **Median (p50)**: ~₹1.24M / day
- **Percentile 90 (p90)**: ~₹2.19M / day

### The "Fuel Gauge" Impact
By implementing **Conformal Quantile Regression (CQR)**, we corrected systematic forecast biases. This structural improvement shifted the performance floor:
- **Before CQR**: The baseline ($\lambda=0$) occasionally realized losses (Worst Day -₹7.6K).
- **After CQR**: The baseline optimizer naturally identifies and avoids tail risk, resulting in a **Worst Day of +₹51K**.

---

## 📋 Methodology Recap
1. **Decision Window**: Stage 1 DAM schedules computed daily using 200 joint-correlated scenarios.
2. **Realization**: DAM schedules are fixed. RTM dispatch is optimized against actual realized prices.
3. **Costs**: All figures include:
   - ₹200/MWh per side IEX Transaction Fees (₹400/MWh round-trip)
   - ₹650/MWh throughput-based degradation cost
   - ₹50/MWh VOM
   - Round-trip efficiency: 90% ($\eta = 94.87\%$ each direction)
4. **Asset**: 50MW / 200MWh (SoC range: 20–180 MWh, 160 MWh usable)
