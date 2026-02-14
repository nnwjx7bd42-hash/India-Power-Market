# BESS Arbitrage Optimizer — Indian Energy Exchange (IEX)

A **Multi-Day Stochastic Programming** framework for Battery Energy Storage Systems (BESS) trading on the Indian Energy Exchange (IEX). This system optimizes Day-Ahead Market (DAM) and Real-Time Market (RTM) positions across multi-day horizons to maximize risk-adjusted revenue.

---

## Performance Summary (143 Days, Feb–Jun 2025)

| Strategy | Net Revenue (₹M) | Avg Daily (₹K) | Worst Day (₹K) | vs Baseline |
| :--- | ---: | ---: | ---: | :---: |
| **Soft Terminal + SoC Chaining** | **201.8** | **1,411** | **+55** | **Baseline** |
| 48h Rolling Horizon (Option B) | 202.8 | 1,418 | −104 | −0.8% |
| 48h Rolling Horizon (Option A) | 196.9 | 1,377 | −104 | −3.7% |
| 7-Day Extensive Form (Option B) | 200.6 | 1,403 | −121 | −1.9% |
| 7-Day Extensive Form (Option A) | 195.2 | 1,365 | −111 | −4.5% |
| Perfect Foresight (Ceiling) | 251.0 | — | — | — |

**Capture Ratio**: 80.4% of perfect-foresight revenue (net-cost basis, soft terminal).

> **Option A** = Hard terminal (SoC must return to 100 MWh each day).  
> **Option B** = Physical floor only (SoC allowed to drop to 20 MWh).

---

## Technical Context

The Indian power market features sequential DAM and RTM settlement. A merchant BESS can arbitrage these spreads, but faces significant **forecast uncertainty** and **RTM tail risk**. This system solves the dispatch problem through:

1. **Probabilistic Forecasting**: LightGBM Quantile Regression predicts the full price distribution (q10–q90) for both Day D and Day D+1.
2. **Joint Scenario Generation**: A Gaussian Copula samples 200 correlated DAM/RTM price paths with cross-day temporal correlation (ρ = 0.24).
3. **Multi-Day Stochastic Optimization**: Solves 48-hour rolling or 7-day extensive-form LPs, committing to DAM schedules (Stage 1) while modeling RTM recourse (Stage 2).
4. **Risk Management**: CVaR with Conformal Quantile Regression (CQR) ensures a secure profit floor — worst-day lifts from −₹97K to **+₹75K** at <1% cost.

---

## System Workflow

```
D-1 08:00 IST Snapshot
    │
    ├─► build_features.py ─────► Day D features + Day D+1 features
    │
    ├─► train_models.py ───────► DAM/RTM quantile models + D+1 DAM model
    │                             (WMAPE: DAM 15.7%, RTM 11.3%, D+1 19.3%)
    │
    ├─► run_recalibration.py ──► CQR conformal shifts (coverage correction)
    │
    ├─► fit_joint_copula.py ───► Cross-market ρ(h) + DAM 24×24 Σ + cross-day ρ
    │
    ├─► regenerate_scenarios.py ► 200 joint DAM/RTM scenarios (single-day)
    │
    ├─► build_multiday_scenarios.py ► 200 × 7-day correlated scenarios
    │                                  (D from model, D+1 from D+1 model,
    │                                   D+2..D+6 climatological fallback)
    │
    └─► Optimizer (choose one):
         ├─ run_phase3b_backtest.py ──── Single-day two-stage SP
         ├─ run_rolling_horizon_backtest.py ── 48h rolling LP
         └─ run_multiday_backtest.py ──── 7-day extensive-form LP
```

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

| Config | Purpose |
| :--- | :--- |
| `bess.yaml` | 50MW / 200MWh nameplate (160MWh usable, 80% DoD), 90% RT efficiency, terminal mode |
| `costs_config.yaml` | ₹200/MWh IEX fees (per side), ₹650/MWh degradation, DSM friction proxy |
| `model_config.yaml` | Quantile levels, LGBM hyperparameters, D+1 forecaster settings |
| `phase3b.yaml` | Single-day optimizer: 200 scenarios, α=0.1 CVaR |
| `phase4_rolling.yaml` | 48h rolling horizon: 50 scenarios, Option A (hard terminal eval) |
| `phase4_rolling_optb.yaml` | 48h rolling horizon: Option B (physical floor eval) |
| `phase4_multiday.yaml` | 7-day extensive form: 30 scenarios, Option A |
| `phase4_multiday_optb.yaml` | 7-day extensive form: Option B |

---

## Key Scripts

| Script | Purpose |
| :--- | :--- |
| `build_features.py` | Feature engineering: Day D + Day D+1 feature sets |
| `train_models.py` | Train DAM, RTM, and D+1 quantile models |
| `fit_joint_copula.py` | Estimate cross-market and cross-day correlations |
| `build_multiday_scenarios.py` | Generate 7-day cross-day correlated scenario sets |
| `run_phase3b_backtest.py` | Single-day two-stage SP backtest |
| `run_rolling_horizon_backtest.py` | 48h rolling horizon backtest (`--config` for Option A/B) |
| `run_multiday_backtest.py` | 7-day extensive form backtest (`--config` for Option A/B) |
| `compare_multiday_strategies.py` | Multi-strategy comparison dashboard (table + charts) |
| `compare_strategies.py` | Baseline vs perfect foresight benchmarking |

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
