# BESS Arbitrage Optimization Stack for the Indian IEX Day-Ahead Market

End-to-end system for **Battery Energy Storage System (BESS) revenue optimization** on India's IEX Day-Ahead Market (DAM). Covers the full pipeline from raw data ingestion through probabilistic price forecasting to stochastic dispatch optimization with realistic merchant cost modelling.

Built for a **20 MW / 40 MWh LFP system** operating under real IEX bidding rules, CERC deviation settlement, and SECI-tender cycle constraints.

---

## Architecture Overview

```
 Raw Data (IEX, NERLDC, Weather)
          |
          v
 ┌─────────────────────────────┐
 │  Layer 1: Data Pipeline     │  src/data_pipeline/
 │  Ingestion, cleaning,       │  ~33k hourly rows, 25+ features
 │  feature engineering        │  Sep 2021 – Jun 2025
 └────────────┬────────────────┘
              |
              v
 ┌─────────────────────────────┐
 │  Layer 2: Point Forecasts   │  v2/
 │  LSTM + XGBoost + Ensemble  │  Optuna-tuned, nested temporal CV
 └────────────┬────────────────┘
              |
              v
 ┌─────────────────────────────┐
 │  Layer 3: Probabilistic     │  v4/
 │  Forecasting Engine         │  Multi-quantile XGBoost (11 τ)
 │  + Conformal Calibration    │  Adaptive Conformal Inference
 │  + Scenario Generation      │  Gaussian Copula → 200 → 10 paths
 └────────────┬────────────────┘
              |
              v
 ┌─────────────────────────────┐
 │  Layer 4: Stochastic        │  v5/
 │  BESS Optimizer             │  CVaR-constrained LP (Pyomo+HiGHS)
 │  + Rolling Backtest         │  Columbia consensus baseline
 └────────────┬────────────────┘
              |
              v
 ┌─────────────────────────────┐
 │  Layer 5: Realistic         │  v6/
 │  Merchant Optimizer         │  IEX fees, CERC DSM, Tx losses
 │  + Cycle limits + Degrada-  │  15-min bid discretisation
 │    tion + Sensitivity       │  75-week rolling backtest
 └─────────────────────────────┘
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for a detailed technical walkthrough of each layer.

---

## Quick Start

### Prerequisites

- Python 3.11+ (developed on 3.14)
- ~2 GB disk for raw data (not included — see Data section)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Pipeline

Each layer has its own entry point:

```bash
# Layer 2 — Train point forecasters
python v2/train_lstm_final.py
python v2/train_xgb_final.py

# Layer 3 — Train quantile model + generate scenarios
python v4/train_quantile_model.py
python v4/run_planning.py

# Layer 4 — Run single-week optimisation
python v5/run_optimizer.py --week 2024-01-01

# Layer 5 — Full realistic backtest
python v6/run_backtest.py --verbose

# Layer 5 — Sensitivity: zero transmission loss
python v6/run_backtest.py --tx-costs-config v6/config/transaction_costs_tx0.yaml

# Layer 5 — Sensitivity: tighter cycle cap (7 cycles/week)
python v6/run_backtest.py --bess-config v6/config/bess_config_cycles7.yaml
```

### Testing

Run the test suite to verify correctness:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_optimizer.py -v
```

The test suite includes:
- **Data pipeline**: Vectorized estimation and aggregation correctness
- **Optimizer**: LP feasibility, SoC bounds, cycle constraint enforcement
- **Backtest**: Look-ahead bias guard (ensures no future data leakage)

---

## Data Requirements

**Raw data is NOT included** in this repository. You need to supply:

| Source | What | Format |
|--------|------|--------|
| [IEX](https://www.iexindia.com/) | Day-Ahead Market clearing prices | Monthly CSVs |
| [NERLDC / POSOCO](https://nerldc.in/) | Regional demand, supply, RE generation | Monthly Excel files |
| [Open-Meteo](https://open-meteo.com/) | Historical weather for 5 Indian cities | API / Parquet cache |

Place files according to the structure in `src/data_pipeline/` loaders, then run the data pipeline scripts to produce the cleaned Parquet dataset consumed by all downstream layers.

The data pipeline (`src/data_pipeline/`) handles ingestion, cleaning, feature engineering, and merging into a unified dataset.

---

## Project Structure

```
├── src/                          # Shared library code
│   ├── data_pipeline/            # Data loaders + feature engineering
│   ├── models/                   # Point forecast architectures (baseline, enhanced, quantile)
│   ├── validation/               # Metrics, temporal CV, diagnostics
│   ├── inference/                # Inference pipeline (feature builder, predictor)
│   └── utils/                    # Calendar feature helpers
│
├── v2/                           # Point forecasting (LSTM + XGBoost)
│   ├── train_lstm.py             # LSTM training with early stopping
│   ├── train_xgb_final.py        # XGBoost final training
│   ├── tune_lstm_bayesian.py     # Optuna Bayesian HP search
│   ├── tune_lstm_nested_cv.py    # Nested temporal cross-validation
│   └── lstm_config.yaml
│
├── v3/                           # Weighted ensemble (LSTM + XGBoost)
│   ├── run_ensemble.py           # Prediction + analysis pipeline
│   ├── ensemble_predict.py       # Load models, combine predictions
│   ├── ensemble_analysis.py      # Error correlation, plots
│   └── ensemble_config.yaml
│
├── v4/                           # Probabilistic forecasting engine
│   ├── config/                   # Planning config (features, quantiles, scenarios)
│   ├── data/                     # Price anchor features + dataset prep
│   ├── models/                   # Multi-quantile XGBoost + conformal wrapper
│   ├── evaluation/               # Pinball loss, CRPS, calibration plots
│   ├── scenarios/                # Gaussian copula + forward reduction
│   ├── train_quantile_model.py   # Train + evaluate quantile forecaster
│   └── run_planning.py           # End-to-end: forecast → calibrate → scenarios
│
├── v5/                           # Stochastic BESS optimizer
│   ├── config/                   # BESS params + optimiser settings
│   ├── optimizer/                # Deterministic LP, Stochastic CVaR LP, Consensus
│   ├── backtest/                 # Rolling backtest harness + baselines + reporting
│   ├── run_optimizer.py          # Single-week runner
│   └── run_backtest.py           # Full rolling backtest
│
├── v6/                           # Realistic merchant BESS optimizer
│   ├── config/                   # BESS, market, transaction costs, sensitivity configs
│   ├── optimizer/                # LP + CVaR with costs, DSM, cycle limits, bid discretisation
│   ├── backtest/                 # Rolling backtest with full cost decomposition
│   ├── run_optimizer.py          # Single-week runner
│   └── run_backtest.py           # Full rolling backtest + sensitivity analysis CLI
│
├── docs/                         # Technical notes
├── tests/                        # Pytest test suite (smoke tests)
│   ├── test_data_pipeline.py     # Data pipeline correctness
│   ├── test_optimizer.py         # Optimizer feasibility & constraints
│   └── test_backtest.py         # Backtest look-ahead bias guard
├── requirements.txt
├── ARCHITECTURE.md               # Detailed technical walkthrough
└── .gitignore
```

---

## Key Design Decisions

### Why no near-term price lags in the forecaster?

The optimizer runs on a **week-ahead planning horizon** — by the time you submit IEX DAM bids (10:00–12:00 the day before), you don't have recent clearing prices. The v4 probabilistic engine uses only exogenous features (weather, load forecasts, calendar) and long-memory price anchors (same hour last week, rolling statistics by hour-of-day).

### Why a pure LP (no binary variables)?

IEX DAM prices are non-negative, so simultaneous charge + discharge is never optimal. This means we can relax the binary mutual-exclusion constraint and solve a continuous LP, which is orders of magnitude faster than MILP. Both the deterministic (perfect foresight) and stochastic (CVaR) formulations exploit this.

### Why CVaR and not just expected value?

Risk-neutral optimisation chases high-variance strategies. The Rockafellar-Uryasev CVaR formulation adds a tail-risk penalty that keeps the optimizer from over-concentrating bets on a few high-spread scenarios. The `beta` parameter controls the risk-aversion dial.

### Why conformal calibration?

Raw quantile regression tends to be miscalibrated — the 90% prediction interval might only contain 82% of actual outcomes. Adaptive Conformal Inference adjusts the intervals online, providing asymptotic coverage guarantees even under distribution shift.

### How is look-ahead bias prevented in the backtest?

At DAM bid-submission time, future hourly demand is unknown. The backtest simulates realistic week-ahead inputs by replacing actual future load features (`Demand`, `Net_Load`, `RE_Penetration`, `Solar_Ramp`) with historical proxies (same-hour-last-week, or 4-week rolling mean by hour-of-day). The model predicts prices using only information available at planning time, while revenue is still computed against actual prices for realistic evaluation.

---

## Looking for Feedback On

This is a solo-built project and I'd genuinely appreciate expert critique on:

1. **Probabilistic forecasting architecture** — Is multi-quantile XGBoost + conformal calibration + Gaussian copula the right stack for week-ahead price scenarios? Should I explore normalizing flows, GAMLSS, or DeepAR instead?

2. **Stochastic optimisation formulation** — Is the Rockafellar-Uryasev CVaR LP the right tool here, or would Stochastic Dual Dynamic Programming (SDDP) or robust optimisation be more appropriate?

3. **Cycle constraint modelling** — I model discharge-only throughput for cycle counting. Is this standard practice, or should I track equivalent full cycles differently?

4. **Indian market cost assumptions** — IEX transaction fees at INR 20/MWh, SLDC at INR 5, RLDC at INR 2, 3% transmission loss, CERC DSM slab structure. Are these realistic for 2024–2025? Would love validation from anyone with actual merchant BESS experience in India.

5. **Backtest methodology** — Rolling weekly with no model re-training during the backtest window. The backtest eliminates look-ahead bias by replacing actual future load values with historical proxies (same-hour-last-week) when building forecast features, while still evaluating against actual prices. Is this sufficient, or should I implement expanding-window re-training?

6. **What's missing** — Intraday re-optimisation? Real-time market (RTM) participation? More sophisticated degradation modelling (rainflow counting vs linear throughput)?

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Point forecasting | PyTorch (LSTM), XGBoost |
| Probabilistic forecasting | XGBoost quantile regression |
| Hyperparameter tuning | Optuna |
| Conformal calibration | Custom ACI implementation |
| Scenario generation | Gaussian copula + forward reduction |
| Optimisation modelling | Pyomo |
| LP solver | HiGHS |
| Data processing | pandas, NumPy, SciPy |
| Visualisation | matplotlib, seaborn |

---

## License

This repository contains source code only. Raw data, trained model weights, and backtest results are excluded and not distributed.
