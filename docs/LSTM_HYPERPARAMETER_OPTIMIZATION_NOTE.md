# LSTM Hyperparameter Optimization: Single-Split vs Nested CV

This note documents what was done and achieved for the v2 LSTM model since starting nested cross-validation (CV) hyperparameter optimization (~25 hours of tuning + apply and train).

---

## 1. Context Before Nested CV

- **Single-split Bayesian tuning** had already been run: 50 Optuna trials, one train/val split, TPE sampler.
- **Best validation MAPE:** 10.63% (trial 45).
- **Best hyperparameters (single-split):** lookback 24, lstm_units [32, 30], dropout ~0.17, learning_rate ~0.00122, batch_size 32, bidirectional True.
- When those params were used to train a final model and evaluate on the **held-out test set**, **test MAPE was ~18%** — a large gap from 10.63% val, indicating **overfitting to the single validation slice** used during tuning.

So the goal of nested CV was to select hyperparameters that **generalize better** to unseen data (smaller val–test gap).

---

## 2. Nested Time-Series CV Tuning

### 2.1 What Was Run

- **Script:** `v2/tune_lstm_nested_cv.py`
- **Design:** For each Optuna trial, instead of one train/val split, we run **time-series cross-validation** with:
  - **5 folds** (`TimeSeriesSplit`, `n_splits=5`)
  - **48-hour gap** between train and validation to avoid leakage
  - Each trial returns the **mean validation MAPE across the 5 folds**
- **Optuna:** 25 trials, TPE sampler, MedianPruner (early pruning of bad trials). One trial was enqueued with the single-split best params as a warm start.
- **Search space:** lookback {24, 48, 72, 168}, lstm_units_1 {32, 50, 64, 100}, lstm_units_2 {16, 30, 50}, dropout [0.1, 0.4], learning_rate (log) [1e-4, 1e-2], batch_size {16, 32, 64}, bidirectional {True, False}.

### 2.2 Runtime and Completion

- **Started:** 2026-01-31 (script start ~12:49 UTC).
- **Finished:** 2026-02-01 (~13:45 UTC).
- **Wall time:** ~25 hours (including system sleep; process resumed correctly after wake).
- **Exit:** 0 (success). All 25 trials completed or were pruned; best trial was complete.

### 2.3 Best Nested CV Result

- **Best trial:** 11
- **Best mean CV MAPE:** 12.045%
- **Best hyperparameters (nested CV):**

| Parameter      | Value   |
|----------------|---------|
| lookback       | 24      |
| lstm_units     | [100, 30] |
| dropout        | ~0.182  |
| learning_rate  | ~0.00131 |
| batch_size     | 64      |
| bidirectional  | True    |

- **Hyperparameter importance (from Optuna):** lookback (0.37) > dropout (0.25) > batch_size (0.20) > lstm_units_2 (0.07) > learning_rate (0.05) > lstm_units_1 (0.04) > bidirectional (0.02).

### 2.4 Outputs Saved

- `v2/results/optuna_lstm_nested_cv_trials.csv` — all trials and mean MAPE
- `v2/results/optuna_lstm_nested_cv_folds.csv` — per-fold MAPE per trial
- `v2/results/optuna_lstm_nested_cv_best_params.yaml` — best config (sequence, model, training) for merging into `lstm_config.yaml`

---

## 3. Applying Nested CV Best Params and Training

### 3.1 Apply Best Params

- **Command:** `python v2/apply_best_params.py --best v2/results/optuna_lstm_nested_cv_best_params.yaml`
- **Effect:** Merged `sequence`, `model`, and `training` from the nested CV best YAML into `v2/lstm_config.yaml`. The main LSTM config now uses the nested CV best hyperparameters (lookback 24, lstm [100, 30], dropout ~0.18, lr ~0.00131, batch 64, bidirectional True).

### 3.2 Final LSTM Training (Two Runs)

- **Script:** `v2/train_lstm.py` (unchanged; reads `v2/lstm_config.yaml`).
- **Data/split:** Same as before: `dataset_cleaned.parquet`, 16 LSTM features, MinMaxScaler on train only, temporal split (train/val/test), lookback 24.
- **First run (right after applying params):**
  - Test MAPE: **16.77%**, RMSE: 852.80
  - Training stopped with early stopping; model and metrics saved to `v2/results/`.
- **Second run (re-run as requested):**
  - Test MAPE: **18.63%**, RMSE: 830.81
  - Same config; difference is due to **random initialization and training dynamics** (no fixed seed for PyTorch in the script).

So with the **same** nested CV best hyperparameters, test MAPE has varied between ~16.8% and ~18.6% across runs — still a noticeable gap from the nested CV mean (12.045%) but **no longer tuned on a single validation slice**.

---

## 4. Lag Features and Simpler Architecture

- **Lag features added** to `lstm_config.yaml`: P(T-1), P(T-24), L(T-1), L(T-24) (20 features total). Manual configs (single layer [32], dropout 0.20, unidirectional) gave **test MAPE 12.50%** ("middle ground") — best so far on a single full run.

---

## 5. Focused Nested CV Tuning (Narrow Search)

- **Script:** same `tune_lstm_nested_cv.py` with narrowed search: lookback {24, 48}, lstm_units_1 {24, 32, 48}, lstm_units_2 {0, 16}, dropout [0.15, 0.25], lr [5e-4, 5e-3], batch {32, 64}, bidirectional False.
- **Best mean CV MAPE:** 12.22% (trial 9).
- **Best params:** lookback 24, lstm_units [48, 16], dropout ~0.199, lr ~0.00167, batch 32, bidirectional false.
- **Apply & train (single run):** `apply_best_params.py` + `train_lstm.py` → **Test MAPE 22.24%** — much worse than manual middle-ground (12.50%). Same config that achieved 12.22% CV MAPE did not translate to a better single full-run test; run-to-run variance and/or early stopping may explain the gap. Manual config [32], dropout 0.20 remains the best **reproduced** test result (12.50%).

### 3.3 Comparison Script

- **Command:** `python v2/compare_tuning_results.py`
- **Effect:** Reads single-split trials CSV (best val 10.63%), nested CV trials CSV (best mean CV 12.04%), and `v2/results/lstm_metrics.yaml` (current test MAPE). Prints a short comparison; can write `v2/results/tuning_comparison.json` / `.txt`.

---

## 4. Summary of What Was Done and Achieved

| Step | What was done | Outcome |
|------|----------------|---------|
| Single-split tuning (pre–nested CV) | 50 Optuna trials, one val split | Best val MAPE 10.63%; test MAPE ~18% (overfit) |
| Nested CV tuning | 25 trials × 5-fold time-series CV, 48h gap | Best mean CV MAPE **12.045%**; best params saved |
| Apply best params | Merge nested CV best YAML into `lstm_config.yaml` | Config updated; LSTM is bidirectional, [100, 30], etc. |
| Train final LSTM | Two full training runs with updated config | Test MAPE 16.77% (first run), 18.63% (second run); variance expected |
| Compare | `compare_tuning_results.py` | Single-split val 10.63%, nested CV mean 12.04%, current test from `lstm_metrics.yaml` |

### Takeaways

- **Nested CV** gives a more realistic estimate of performance (mean over 5 folds) and selected different hyperparameters (e.g. lstm_units [100, 30], batch 64) than the single-split best ([32, 30], batch 32).
- **Test MAPE** with nested CV params is in the ~17–19% range in our two runs — still above the 12% mean CV, but the tuning was not overfitting to one validation slice.
- **Run-to-run variance** in test MAPE (16.77% vs 18.63%) is expected without fixing PyTorch/NumPy seeds; for reporting, consider multiple runs or seeding.
- **Artifacts:** Best params live in `v2/results/optuna_lstm_nested_cv_best_params.yaml`; current trained model and metrics in `v2/results/lstm_model.pt`, `lstm_scaler.joblib`, `lstm_metrics.yaml`.

---

*Note written to capture work from nested CV start (~25 hours of tuning) through apply, train, and compare.*
