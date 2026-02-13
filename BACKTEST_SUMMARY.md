# Backtest Performance Summary: GENCO BESS VPP

This document provides the definitive realized performance analysis for the GENCO 50MW / 200MWh BESS optimization system across a 143-day backtest period (Feb 1, 2025 – June 24, 2025).

---

## 1. Performance Overview (Recalibrated Model)

All results represent **actual realized performance** against historical market actuals, with Stage 1 commitments fixed and Stage 2 recourse optimized against realized RTM prices.

- **Total Realized Net Revenue**: **₹198,073,460**
- **Average Daily Revenue**: ₹1,385,129
- **Median Day Revenue**: ₹1,242,316
- **Performance Floor (Worst Day)**: **+₹50,978** (Resilience via CQR)
- **Unit Economics (Normalized)**: ~₹2.5M / MWh-installed / year

> [!IMPORTANT]
> **Seasonality Caveat**  
> The Feb–June backtest window corresponds to India's peak market volatility and high price spreads. These results should not be linearly extrapolated to the full fiscal year, as monsoon and shoulder months typically exhibit narrower arbitrage windows.

---

## 2. Benchmarking Analysis

To evaluate the efficiency of the Two-Stage Stochastic Program, we compare the recalibrated model against both an upper-bound "Perfect Foresight" scenario and a naive "Deterministic" baseline.

| Strategy | Net Revenue (₹M) | % of Perfect Foresight | Worst-Day Profile |
| :--- | :--- | :--- | :--- |
| **Perfect Foresight (Ceiling)** | **239.70** | 100% | N/A |
| **Stochastic SP (Recalibrated)** | **198.07** | **82.6%** | **Positive (+₹51K)** |
| **Deterministic (q50 Forecast)** | 154.60 | 64.5% | Negative (-₹12K) |

**Key Finding**: The recalibrated Stochastic system captures **82.6% of theoretical maximum returns**, outperforming the naive deterministic baseline by **₹43.4M (+28%)** while maintaining a safe performance floor.

---

## 3. Financial Waterfall (Actuals)
*Aggregate values over the 143-day period at $\lambda=0$.*

| Component | Value (₹M) | Description |
| :--- | :--- | :--- |
| **Gross Arbitrage Revenue** | **226.75** | Market-clearing realized income. |
| (-) IEX Transaction Fees | (14.28) | ₹200/MWh per side on physical churn. |
| (-) Degradation Costs | (12.45) | ₹650/MWh based on discharge throughput. |
| (-) Variable O&M | (1.95) | ₹50/MWh estimated operating cost. |
| **Final Net Revenue** | **198.07** | **Realized bottom-line performance.** |

---

## 4. Risk-Return Frontier (Recalibrated System)

The following table demonstrates the impact of the risk-aversion coefficient ($\lambda$) on realized outcomes.

| Lambda ($\lambda$) | Net Revenue (₹M) | Worst Day (₹K) | Custom Resilience* | Avg Daily Cycles |
| :--- | :--- | :--- | :--- | :--- |
| **0.00 (Baseline)** | **198.07** | **+50.98** | 1.04 | 1.20 |
| **0.10 (Balanced)** | 195.26 | +63.63 | 1.05 | 1.20 |
| **0.50 (Defensive)** | 187.09 | **+138.72** | **1.12** | 1.20 |

*\*Internal Metric: Custom Resilience = Mean / (Mean - Worst). Measures the strength of the profit floor relative to average returns.*

---

## 5. Analytical Conclusion
The backtest results confirm that the transition from deterministic to stochastic modeling — reinforced by Conformal Quantile Regression — provides the optimal risk-adjusted strategy for Indian BESS assets. The system effectively filters out RTM tail-risk (recovering **+₹51K** on the worst day where forecasts initially suggested a loss) without significantly degrading capital efficiency.
