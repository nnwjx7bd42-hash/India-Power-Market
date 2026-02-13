# BESS Virtual Power Plant (VPP) Optimizer

A state-of-the-art Stochastic Optimization system for Battery Energy Storage Systems. This build implements a cost-aware, risk-managed trading agent for the Indian Power Market (IEX).

## 📋 Table of Contents
1. [Installation](#-installation)
2. [Data Pipeline](#-data-pipeline)
3. [Model Training](#-model-training)
4. [Optimization & Backtesting](#-optimization--backtesting)
5. [Configuration](#-configuration)
6. [Advanced: Recalibration (CQR)](#-advanced-recalibration-cqr)

---

## 🛠 Installation

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

## 📈 Data Pipeline

The system expects data in the `Data/` directory:
- `Data/Raw/`: CSVs of historical MCP (Market Clearing Price).
- `Data/Features/`: Precompute features using:
  ```bash
  python scripts/build_features.py
  ```

---

## 🧠 Model Training

We use LightGBM Quantile Regression to predict the price distribution (q10, q25, q50, q75, q90).
```bash
python scripts/train_models.py
```
*Models are saved in `models/rtm/` and `models/dam/`.*

---

## ⚖️ Optimization & Backtesting

### 1. Standard Two-Stage Backtest
Calculates Stage 1 commitments and solves for Stage 2 recourse against actual realized prices.
```bash
python scripts/run_phase3b_backtest.py
```

### 2. CVaR Risk-Return Sweep
Maps the **Efficient Frontier** by varying the risk-aversion coefficient ($\lambda$).
```bash
# Uses the original scenario fan
python scripts/run_cvar_sweep.py --scenarios original

# Uses the CQR-recalibrated fan (Phase 4B outcome)
python scripts/run_cvar_sweep.py --scenarios recalibrated
```

---

## ⚙️ Configuration

- `config/bess.yaml`: Physical battery specs (P_max, E_max, Efficiency, Degradation Cost).
- `config/phase3b.yaml`: Optimizer settings (n_scenarios, deviation penalties).
- `config/cvar_config.yaml`: Risk settings (Alpha, Lambda values).

---

## 🎯 Advanced: Recalibration (CQR)

If you notice "prediction gaps" (coverage mismatch), run the CQR engine:
1. **Compute Deltas**: 
   ```bash
   python scripts/run_recalibration.py
   ```
2. **Regenerate Scenarios**:
   ```bash
   python scripts/regenerate_scenarios.py
   ```

---

## 📁 Directory Structures

```text
.
├── src/                # Implementation logic
│   ├── features/       # Feature engineering & IEX lags
│   ├── forecasting/    # LGBM Quantile Models & CQR engine
│   ├── optimizer/      # PuLP SLP formulation
│   └── scenarios/      # Copula & Joint generation
├── scripts/            # Entry point for all pipelines
├── config/             # YAML specification files
├── models/             # Serialized .txt model artifacts
└── results/            # Performance JSONs & Parquet logs
```

---

## 🛡 License & Authors
Project Alpha - Clean Build VPP. 
Designed for Advanced BESS Arbitrage.
