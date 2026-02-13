# BESS Virtual Power Plant (VPP) Optimizer

This repository contains a Two-Stage Stochastic Programming framework for optimizing Battery Energy Storage Systems (BESS) within the Indian Energy Exchange (IEX).

## Executive Context
In the Indian power market, the coexistence of the Day-Ahead Market (DAM) and Real-Time Market (RTM) presents a significant arbitrage opportunity for energy storage. However, naive "spread-chasing" strategies often fall victim to forecast uncertainty and "bull traps" in the RTM tail. 

This system solves the arbitrage problem by modeling it as a **Two-Stage Stochastic Program (SP)**. It commits to DAM schedules while accounting for 200 possible RTM scenario paths, explicitly incorporating Conditional Value at Risk (CVaR) to ensure commercial resilience against tail-end price spikes or crashes.

## Table of Contents
1. [Installation](#installation)
2. [Data Pipeline](#data-pipeline)
3. [Model Training](#model-training)
4. [Optimization & Backtesting](#optimization--backtesting)
5. [Configuration](#configuration)
6. [Advanced: Recalibration (CQR)](#advanced-recalibration-cqr)

---

## Installation

### 1. Prerequisites
- Python 3.9+
- **CBC Solver**: The optimizer uses `PuLP` with the `CBC` solver by default. 
  - Mac: `brew install cbc`
  - Linux: `sudo apt-get install cbc`

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data Pipeline

The system expects data in the `Data/` directory:
- `Data/Raw/`: CSVs of historical market clearing prices (MCP).
- `Data/Features/`: Generate technical and market features using:
  ```bash
  python scripts/build_features.py
  ```

---

## Model Training

We use LightGBM Quantile Regression to predict the price distribution (q10, q25, q50, q75, q90).
```bash
python scripts/train_models.py
```
*Models are serialized to `models/rtm/` and `models/dam/`.*

---

## Optimization & Backtesting

### 1. Two-Stage Backtest
Calculates Stage 1 (DAM) commitments and solves for Stage 2 (RTM) recourse against actual realized market prices.
```bash
python scripts/run_phase3b_backtest.py
```

### 2. Risk-Return Analysis
Maps the **Efficient Frontier** by varying the risk-aversion coefficient ($\lambda$).
```bash
# Evaluate against original scenarios
python scripts/run_cvar_sweep.py --scenarios original

# Evaluate against CQR-recalibrated scenarios
python scripts/run_cvar_sweep.py --scenarios recalibrated
```

---

## Configuration

- `config/bess.yaml`: Physical asset specifications (50MW / 200MWh, 90% RTE).
- `config/phase3b.yaml`: Optimizer settings (200 scenarios, deviation penalties).
- `config/costs_config.yaml`: Market friction and asset costs (₹200/side IEX Fees, ₹650 Degradation).

---

## Advanced: Recalibration (CQR)

If the realized coverage (e.g., actuals falling within the q10-q90 interval) deviates significantly from the target (e.g., >5% error), run the Conformal Quantile Regression (CQR) engine:
1. **Compute Residual Deltas**: 
   ```bash
   python scripts/run_recalibration.py
   ```
2. **Regenerate Calibrated Scenarios**:
   ```bash
   python scripts/regenerate_scenarios.py
   ```

---

## Results Summary
- **Performance**: Normalized net revenue of ~₹2.5M / MWh-cap / year.
- **Resilience**: CQR recalibration provides a secure profit floor of **+₹51K Worst Day** at $\lambda=0$.
- **Asset Specs**: 50MW / 200MWh BESS with 90% Round-trip efficiency.

---

## Project Status
**Authorship**: GENCO Clean Build VPP. 
Designed for institutional-grade BESS arbitrage and market risk management.
