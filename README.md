# BESS Arbitrage Optimizer — Indian Energy Exchange (IEX)

A **Two-Stage Stochastic Programming** framework for Battery Energy Storage Systems (BESS) trading on the Indian Energy Exchange (IEX). This system optimizes Day-Ahead Market (DAM) and Real-Time Market (RTM) positions to maximize risk-adjusted revenue.

---

## Technical Context

The Indian power market features sequential DAM and RTM settlement. A merchant BESS can arbitrage these spreads, but faces significant **forecast uncertainty** and **RTM tail risk**. This system solves the dispatch problem by:
1.  **Probabilistic Forecasting**: Using LightGBM Quantile Regression to predict the full price distribution (q10–q90).
2.  **Joint Scenario Generation**: Using a Gaussian Copula to sample 200 correlated price paths that preserve cross-market dependencies.
3.  **Two-Stage Stochastic Optimization**: Committing to DAM schedules (Stage 1) while modeling RTM recourse (Stage 2) across all 200 scenarios.
4.  **Risk Management**: Incorporating Conditional Value at Risk (CVaR) with Conformal Quantile Regression (CQR) to ensure a secure profit floor.

**Result**: The optimizer captures **78.9% of perfect-foresight revenue** (net-cost basis), providing institutional-grade performance of ~₹1.7M/MWh-cap/year with a confirmed profit floor.

---

## Performance Summary (143 Days)

*Training Period: Apr 1, 2022 – Sep 30, 2024 (Historical Market Clearing)*  
*Validation Period: Oct 1, 2024 – Jan 31, 2025 (Out-of-sample Calibration)*  
*Backtest Period: Feb 1, 2025 – Jun 24, 2025 (Peak Spread Season).*

| Metric | Stochastic SP (Recalibrated) | Deterministic (q50) | Perfect Foresight |
| :--- | :--- | :--- | :--- |
| **Total Net Revenue** | **₹198.1M** | ₹198.3M | ₹251.0M |
| **Capture Ratio** | **78.9%** | 79.0% | 100% |
| **Worst-Day Outcome** | **+₹3K** | -₹25K | N/A |
| **Avg. Daily Cycles** | 1.2 | 1.1 | 1.3 |

**Risk-Return Frontier ($\lambda$ Sweep)** *(post-regulatory net, CERC 2024)*:
- **Baseline ($\lambda=0$):** ₹196.3M net, +₹75K worst-day, Resilience 1.06.
- **Balanced ($\lambda=0.1$):** ₹195.9M net, +₹87K worst-day, Resilience 1.07.
- **Defensive ($\lambda=0.5$):** ₹189.5M net, +₹85K worst-day, Resilience 1.07.

---

## System Workflow

1.  **Data Processing**: Historical MCP data is transformed into technical and market features (`scripts/build_features.py`).
2.  **Probabilistic Modeling**: LightGBM models are trained on five quantiles to capture price uncertainty (`scripts/train_models.py`).
3.  **CQR Recalibration**: Conformal Quantile Regression shifts quantiles to ensure empirical coverage matches nominal targets (`scripts/run_recalibration.py`).
4.  **Scenario Generation**: 200 joint DAM/RTM scenarios are sampled using a Gaussian Copula (`scripts/regenerate_scenarios.py`).
5.  **Optimization**: The Two-Stage SP is solved via PuLP (CBC/HiGHS) to find the optimal DAM commitment (`scripts/run_cvar_sweep.py`).

---

## Data Inputs & Features

The LightGBM forecasting models ingest a diverse feature set to capture the complexities of the Indian Power Grid:

- **Market Data (IEX)**: Lagged MCPs for DAM/RTM, DAM-RTM price spreads, and historical bid-stack volumes (Buy/Sell ratios).
- **Grid Fundamentals**: Total system demand, renewable generation (Solar/Wind), net demand, and estimated thermal plant utilization.
- **Weather Intelligence**: Weighted national temperatures, regional Delhi/Chennai data, solar irradiance (shortwave flux), and Cooling Degree Hours (CDH) as a proxy for cooling load.
- **Temporal/Calendar**: Hour of day, day of week, monsoon season flags, and holiday proximity.

---

## Configuration

Core parameters are managed in `config/`:
- **Asset (`bess.yaml`)**: 50MW / 200MWh capacity, 90% Round-trip efficiency.
- **Costs (`costs_config.yaml`)**: ₹200/MWh IEX fees (per side), ₹650/MWh degradation.
- **Optimizer (`phase3b.yaml`)**: 200 scenarios, $\alpha=0.1$ for CVaR tail risk.

---

## Installation

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# System dependency: CBC Solver
# Mac: brew install cbc | Linux: sudo apt-get install cbc
```

---

## Project Status
**Authorship**: GENCO Clean Build VPP. 
Designed for institutional-grade BESS arbitrage and market risk management.
