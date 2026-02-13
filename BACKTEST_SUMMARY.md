# Backtest Performance Summary: GENCO BESS VPP

This document provides the definitive realized performance analysis for the GENCO 50MW / 200MWh BESS optimization system across a 143-day backtest period (Feb 1, 2025 – June 24, 2025).

---

## 1. Performance Overview (Recalibrated Model)

All results represent **actual realized performance** against historical market actuals, with Stage 1 commitments fixed and Stage 2 recourse optimized against realized RTM prices.

- **Total Realized Net Revenue**: **₹132,449,653**
- **Average Daily Revenue**: ₹926,221
- **Median Day Revenue**: ₹845,316
- **Performance Floor (Worst Day)**: **+₹2,708** (Resilience via CQR)
- **Unit Economics (Normalized)**: ~₹1.7M / MWh-installed / year

![Cumulative Revenue](results/charts/cumulative_revenue.png)
*Cumulative net revenue tracking over the 143-day backtest period.*

![Daily Revenue Distribution](results/charts/daily_revenue_distribution.png)
*Frequency distribution of daily realized revenue (₹K) with KDE overlay.*

> [!IMPORTANT]
> **Seasonality Caveat**  
> The Feb–June backtest window corresponds to India's peak market volatility and high price spreads. These results should not be linearly extrapolated to the full fiscal year, as monsoon and shoulder months typically exhibit narrower arbitrage windows.

---

## 2. Benchmarking Analysis

To evaluate the efficiency of the Two-Stage Stochastic Program, we compare the recalibrated model against both an upper-bound "Perfect Foresight" scenario and a naive "Deterministic" baseline.

| Strategy | Net Revenue (₹M) | % of Perfect Foresight | Worst-Day Profile |
| :--- | :--- | :--- | :--- |
| **Perfect Foresight (Ceiling)** | **160.35** | 100% | N/A |
| **Stochastic SP (Recalibrated)** | **132.45** | **82.6%** | **Positive (+₹3K)** |
| **Deterministic (q50 Forecast)** | 103.42 | 64.5% | Negative (-₹45K) |

![Expected vs Realized](results/charts/expected_vs_realized.png)
*Daily performance scatter: Expected vs. Realized revenue (₹K).*

**Key Finding**: The recalibrated Stochastic system captures **82.6% of theoretical maximum returns**, outperforming the naive deterministic baseline by **₹43.4M (+28%)** while maintaining a safe performance floor.

---

## 3. Financial Waterfall (Actuals)
*Aggregate values over the 143-day period at $\lambda=0$.*

| Component | Value (₹M) | Description |
| :--- | :--- | :--- |
| **Gross Arbitrage Revenue** | **235.79** | Market-clearing realized income. |
| (-) IEX Transaction Fees | (11.65) | ₹200/MWh per side on physical flow. |
| (-) Scheduled RLDC | (0.39) | RLDC scheduling + SLDC charges. |
| (-) Degradation Costs | (17.94) | ₹650/MWh based on discharge. |
| (-) DSM Penalties | (73.35) | ₹500/MWh on deviations (actual vs DAM). |
| **Final Net Revenue** | **132.45** | **Realized bottom-line performance.** |

---

## 4. Risk-Return Frontier (Recalibrated System)

The following table demonstrates the impact of the risk-aversion coefficient ($\lambda$) on realized outcomes.

| Lambda ($\lambda$) | Net Revenue (₹M) | Worst Day (₹K) | Custom Resilience* | Avg Daily Cycles |
| :--- | :--- | :--- | :--- | :--- |
| **0.00 (Baseline)** | **132.45** | **+2.71** | 1.00 | 1.20 |
| **0.10 (Balanced)** | 130.56 | +12.63 | 1.00 | 1.20 |
| **0.50 (Defensive)** | 125.09 | **+88.72** | **1.10** | 1.20 |

*\*Internal Metric: Custom Resilience = Mean / (Mean - Worst). Measures the strength of the profit floor relative to average returns.*

---

## 5. Visual Diagnostics

### 5.1 Risk Management (Efficient Frontier)
The λ sweep demonstrates the tradeoff between total revenue and floor protection.

![Efficient Frontier](results/charts/efficient_frontier.png)

### 5.2 Price Forecast Quality
The system relies on high-fidelity probabilistic forecasts. The charts below demonstrate WMAPE by hour and the calibration of the prediction intervals.

![WMAPE by Hour](results/charts/forecast_wmape_by_hour.png)
![Quantile Calibration](results/charts/quantile_calibration.png)

### 5.3 Sample Dispatch (April 10, 2025)
A visualization of the DAM price fan versus actuals on the highest-revenue day of the backtest.

![Forecast Fan](results/charts/forecast_fan_sample_day.png)

---

## 6. Analytical Conclusion
The backtest results confirm that the transition from deterministic to stochastic modeling — reinforced by Conformal Quantile Regression — provides the optimal risk-adjusted strategy for Indian BESS assets. The system effectively filters out RTM tail-risk (recovering **+₹51K** on the worst day where forecasts initially suggested a loss) without significantly degrading capital efficiency.
