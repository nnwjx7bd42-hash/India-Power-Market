# Baseline and Enhanced Models: Data, Processing, Learning, and Evaluation

This note describes what data both models ingest, how they process it, what they do, what they learn on, and what they are tested against to assess learning.

---

## 1. Data Ingestion (Common to Both Models)

Both the **baseline** and **enhanced** models read the **same** training dataset. No separate raw ingestion happens inside the training scripts.

### 1.1 Primary Input File

| Item | Value |
|------|--------|
| **File** | `data/processed/dataset_cleaned.parquet` |
| **Format** | Parquet (columnar) |
| **Index** | Datetime, IST (+05:30) |
| **Date range** | 2021-09-01 00:00:00 IST to 2025-06-24 23:00:00 IST |
| **Granularity** | Hourly (one row per hour) |
| **Typical shape** | ~33,409 rows × 65 columns (64 features + 1 target) |

### 1.2 Upstream Creation of This Dataset

`dataset_cleaned.parquet` is **not** built by the training scripts. It is produced by the data pipeline (e.g. `regenerate_full_dataset.py`), which:

1. **Unified dataset** – Merges IEX day-ahead price (`P(T)`) with NERLDC SCADA (Demand, Thermal, Hydro, Gas, Nuclear, Wind, Solar). Hydro/Gas/Nuclear for 2024–2025 are estimated in the NERLDC load step where needed.
2. **Weather merge** – Merges national weather (Open-Meteo, 5 cities) via `merge_asof`; fills missing weather if any.
3. **Feature engineering** – Adds calendar, weather-derived (CDH, HDH, etc.), supply-derived (RE_Generation, Net_Load, etc.), lag features (price/load lags, rolling stats), and interaction terms.
4. **Cleaning** – Drops redundant/low-correlation features, fills missing lag values (e.g. with mean), and ensures no critical NaNs. Output is saved as `dataset_cleaned.parquet`.

So **what the models “ingest”** is this single, already-merged, feature-engineered, cleaned table. Training code only does:

- `pd.read_parquet('data/processed/dataset_cleaned.parquet')`
- Use the datetime index for temporal splits.

### 1.3 Target Variable

- **Target** for both models: **`P(T)`** — hourly electricity price (₹/MWh) for the same hour as the row.

### 1.4 Configuration Used at Load Time

- **Config file**: `config/model_config.yaml`
- From it, training uses in particular:
  - **CV**: `n_splits`, `gap_hours`, `test_months`
  - **Baseline/Enhanced**: XGBoost hyperparameters (e.g. `max_depth`, `learning_rate`, `n_estimators`).

### 1.5 Training vs. Test Periods: What Data Is Actually Used

With the **current config** (`n_splits: 14`, `holdout_hours: 168`), the pipeline uses **almost all** of the cleaned dataset in temporal cross-validation and reserves the last 168 hours (7 days) for a separate inference test.

- **Cleaned data span**: 2021-09-01 to 2025-06-24 (~33,409 hours).
- **CV setup**: 14 folds (or auto from data if `n_splits` is null), 3 months per test fold, 24-hour gap between train end and test start. Test start dates are every 3 months beginning 4 months after the dataset start (first test: 2022-01-01).
- **Holdout**: If `holdout_hours` > 0 (e.g. 168), the dataframe is trimmed to `index <= max(index) - holdout_hours` before building CV, so the last N hours are never used for training or testing and can be used for inference evaluation.

**Per fold:**

- **Training**: From **2021-09-01** up to **(test_start − 24 hours)**. Each fold uses only the past before that fold’s test window (expanding window).
- **Test**: A contiguous **3-month** block immediately after the 24-hour gap (e.g. 2022-01-01 to 2022-03-31 for fold 1).

**Concrete 14-fold layout (data through 2025-06-24; no holdout or after trim):**

| Fold | Train period (used for fitting) | Test period (used only for evaluation) |
|------|----------------------------------|----------------------------------------|
| 1    | 2021-09-01 → 2021-12-31 (−24h)   | 2022-01-01 → 2022-03-31                |
| 2    | 2021-09-01 → 2022-03-31 (−24h)   | 2022-04-01 → 2022-06-30                |
| 3    | 2021-09-01 → 2022-06-30 (−24h)   | 2022-07-01 → 2022-09-30                |
| 4    | 2021-09-01 → 2022-09-30 (−24h)   | 2022-10-01 → 2022-12-31                |
| 5    | 2021-09-01 → 2022-12-31 (−24h)   | 2023-01-01 → 2023-03-31                |
| 6    | 2021-09-01 → 2023-03-31 (−24h)   | 2023-04-01 → 2023-06-30                |
| 7    | 2021-09-01 → 2023-06-30 (−24h)   | 2023-07-01 → 2023-09-30                |
| 8    | 2021-09-01 → 2023-09-30 (−24h)   | 2023-10-01 → 2023-12-31                |
| 9    | 2021-09-01 → 2023-12-31 (−24h)   | 2024-01-01 → 2024-03-31                |
| 10   | 2021-09-01 → 2024-03-31 (−24h)   | 2024-04-01 → 2024-06-30                |
| 11   | 2021-09-01 → 2024-06-30 (−24h)   | 2024-07-01 → 2024-09-30                |
| 12   | 2021-09-01 → 2024-09-30 (−24h)   | 2024-10-01 → 2024-12-31                |
| 13   | 2021-09-01 → 2024-12-31 (−24h)   | 2025-01-01 → 2025-03-31                |
| 14   | 2021-09-01 → 2025-03-31 (−24h)   | 2025-04-01 → 2025-06-24 (or to holdout start) |

**Summary:**

- **Training data**: From **2021-09-01** up to the end of the **last** fold’s training window. With 14 folds and no holdout, the last fold trains on approximately **2021-09-01 to 2025-03-30** (~32,200 hours). With `holdout_hours: 168`, the dataframe is trimmed so the effective end is 7 days before the file end; the last fold’s train/test then use that trimmed range.
- **Test data**: The 14 three-month blocks above (2022 Q1 through 2025 Q2). The last fold’s test period runs to the end of the (possibly trimmed) data.
- **Unused in CV**: With `holdout_hours: 168`, the **last 168 hours** of the cleaned file are not used in any fold and are reserved for inference test. With `holdout_hours: 0`, there is no reserved period; all data in the file is used in train or test across the 14 folds.

So both models are trained on an **expanding slice** of the past, with the **maximum** training period now ending around **2025-03-30** (or 7 days before file end if holdout is enabled). Setting `n_splits: 14` (and optionally `holdout_hours: 168`) ensures the pipeline uses the full period through 2025 and reserves a short holdout for final evaluation.

---

## 2. Baseline Model

### 2.1 What Data It Uses (Subset of the 65 Columns)

The baseline uses a **fixed subset of 13 features** from the same parquet:

- **Price lags**: `P(T-1)`, `P(T-2)`, `P(T-24)`, `P(T-48)`
- **Load lags**: `L(T-1)`, `L(T-2)`, `L(T-24)`, `L(T-48)`
- **Calendar (raw)**: `Day` (weekend indicator), `Season`, `Hour`, `DayOfWeek`, `Month`

Feature selection is done by `get_baseline_features(df)` in `src/models/baseline.py`: it takes the union of the configured “original” and “calendar” feature names and keeps only those that exist in `df`. So the baseline **ingests** the full table but **uses only these 13 columns** as inputs and **`P(T)`** as the target.

### 2.2 How It Processes the Data

1. **Load**  
   - Read `data/processed/dataset_cleaned.parquet` and `config/model_config.yaml`.

2. **Feature matrix**  
   - `X = df[baseline_features].copy()`  
   - `y = df['P(T)'].copy()`  
   - If any missing values appear in `X`, the baseline code fills them with the column median (per `baseline.py`).

3. **Temporal cross-validation (same for both models)**  
   - **TemporalCV** (in `src/validation/temporal_cv.py`):
     - **Expanding window**: train set always starts at the global start date and grows; test set is a future contiguous block.
     - **Splits**: 7 folds (`n_splits=7`).
     - **Test window**: 3 months per fold (`test_months=3`).
     - **Gap**: 24 hours between end of train and start of test (`gap_hours=24`) to reduce leakage.
   - First test fold starts after ~4 months of data; then every 3 months a new test fold.

4. **Train/validation split (within each fold)**  
   - From the **training indices** of the fold:
     - **Train (fit)**: all but the last 20% of the training period.
     - **Validation**: last 20% of the training period.  
   - Used for XGBoost training and early-stopping monitoring (eval_set).

5. **Model**  
   - **Algorithm**: XGBoost regressor (`reg:squarederror`).  
   - **Hyperparameters**: from `config['baseline']` (e.g. `max_depth=6`, `learning_rate=0.05`, `n_estimators=500`, `subsample=0.8`, `colsample_bytree=0.8`). No tuning step; fixed config.

6. **Training**  
   - For each fold: fit on `(X_train_fit, y_train_fit)` with `eval_set=[(X_val, y_val)]`.  
   - The model **learns** to map the 13 baseline features to `P(T)`.

7. **Saving**  
   - The **last fold’s** trained model is saved as `models/training/baseline_model.pkl`.

### 2.3 What It Learns On

- **Inputs**: The 13 baseline features above (lagged price/load + calendar).  
- **Target**: `P(T)` (hourly price).  
- **Samples**: Only rows whose timestamps fall in the **training part** of each fold (expanding window). Within that, the 80% “train” part is used for fitting; the 20% “validation” part is used only for monitoring (e.g. early stopping), not for gradient updates.  
- So effectively it **learns on** the (feature, target) pairs from the expanding past, up to the gap before each test window.

### 2.4 What It Is Tested Against (Assessment of Learning)

- **Test data**: For each of the 7 folds, the **test set** is the 3-month block **after** the 24-hour gap. The model never sees these timestamps during training.
- **Predictions**: For each test fold, `y_pred = model.predict(X_test)`.
- **Metrics** (from `validation/metrics.py`) computed per fold and then aggregated:
  - **RMSE** (₹/MWh)  
  - **MAE** (₹/MWh)  
  - **MAPE** (%)  
  - **R²**  
  - **Directional accuracy** (%) — proportion of consecutive hour pairs where the sign of the predicted price change matches the sign of the actual price change.
- **Success criteria** (reported in the script): MAPE &lt; 15%, RMSE &lt; ₹2,000, directional accuracy &gt; 60%.
- **Extra diagnostics**: Error by hour (e.g. evening peak 18–20h), feature importance (XGBoost gain), residual plots. Results and plots are written under `results/training/` and `results/training/diagnostics/`.

So **assessment of learning** is: performance on these **future, unseen 3-month test windows** via RMSE, MAE, MAPE, R², and directional accuracy, plus hourly and residual analyses.

---

## 3. Enhanced Model

### 3.1 What Data It Uses (Full Feature Set)

The enhanced model uses **all columns of the same parquet except the target**:

- **Excluded**: `P(T)` (target only).  
- **Features**: All other columns in `dataset_cleaned.parquet` — typically **64 features**.  
- Selection: `get_enhanced_features(df)` in `src/models/enhanced.py` returns `[c for c in df.columns if c != 'P(T)']`.

So it **ingests** the same `dataset_cleaned.parquet` but uses **every available feature** (price/load lags, rolling stats, weather raw and derived, calendar, cyclic encodings, supply-side raw and derived, interaction terms). No feature subsetting.

### 3.2 How It Processes the Data

1. **Load**  
   - Same as baseline: `data/processed/dataset_cleaned.parquet` and `config/model_config.yaml`.

2. **Feature matrix**  
   - `X = df[enhanced_features].copy()` (64 columns), `y = df['P(T)'].copy()`.  
   - Missing values in `X` are again filled with median (in `enhanced.py`’s `prepare_features`).

3. **Temporal cross-validation**  
   - Same **TemporalCV** as baseline: 7 folds, 3-month test windows, 24-hour gap, expanding train window.

4. **Hyperparameter tuning (before per-fold training)**  
   - **tune_with_baseline_comparison** (in `validation/hyperparameter_tuning.py`):
     - Uses the same temporal CV splits.
     - Random search over XGBoost hyperparameters (e.g. `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`).
     - Optimizes **MAPE** (default).
     - Picks the best parameter set; these are then used for all subsequent enhanced folds.
   - So the enhanced model **processes** the same (X, y) but first **tunes** hyperparameters in a temporally valid way.

5. **Train/validation split (within each fold)**  
   - Same as baseline: last 20% of the training period of the fold = validation; first 80% = fit.

6. **Model**  
   - **Algorithm**: XGBoost regressor, same objective.  
   - **Hyperparameters**: From `config['enhanced']`, **overwritten** by the best params from the tuning step (so the enhanced model is tuned; the baseline is not).

7. **Training**  
   - For each fold: fit enhanced XGBoost on the 64 features with the tuned config.  
   - The model **learns** to map all 64 features to `P(T)`.

8. **Quantile models (after point-forecast folds)**  
   - **Separate step** in `train_enhanced.py`: trains **quantile regression** models (P10, P50, P90) using the **same** 64 features and the same dataset.  
   - Uses the **last** temporal fold’s train/test split; train is again split 80/20 for fit/val.  
   - Three XGBoost-style quantile models (e.g. one per quantile).  
   - Evaluated with pinball loss, coverage, interval width; saved as `models/training/quantile_models.pkl` and predictions to `results/training/quantile_predictions.parquet`.

9. **Comparison with baseline**  
   - After training, the script **compares** baseline vs enhanced using the saved CV results and predictions: metrics, by-hour and by-regime breakdowns, and plots/reports under `results/training/comparison_plots/` and `model_comparison_report.md`.

10. **Saving**  
    - Last fold’s enhanced point model: `models/training/enhanced_model.pkl`.  
    - Quantile models: `models/training/quantile_models.pkl`.  
    - All CSV/parquet outputs under `results/training/`.

### 3.3 What It Learns On

- **Inputs**: All 64 features (lags, weather, calendar, supply, interactions).  
- **Target**: Same `P(T)`.  
- **Samples**: Same temporal logic as baseline — expanding training window per fold, with 80% used for fitting and 20% for validation monitoring.  
- So it **learns on** the same type of (past) data as the baseline, but with the full feature set and **tuned** hyperparameters.  
- Quantile models learn the same 64 features but with quantile objectives (P10, P50, P90).

### 3.4 What It Is Tested Against (Assessment of Learning)

- **Point forecast**: Same as baseline — **7 test folds**, each a 3-month future block after the gap.  
- **Metrics**: Same (RMSE, MAE, MAPE, R², directional accuracy), plus comparison to baseline (e.g. significance tests).  
- **Success criteria** (stricter): MAPE &lt; 10%, RMSE &lt; ₹1,000, R² &gt; 0.90, directional accuracy &gt; 75% (all reported as pass/fail).  
- **Quantile models**: Evaluated on the last fold’s test set (pinball loss, P10–P90 coverage, interval width).  
- **Comparison**: Baseline vs enhanced is assessed over the **same** test folds and same metrics to see whether the extra features and tuning improve out-of-sample performance.

So **assessment of learning** is: out-of-sample performance on the **same temporal test windows** as the baseline, with the same core metrics plus quantile metrics and a formal baseline comparison.

---

## 4. Summary Table

| Aspect | Baseline | Enhanced |
|--------|----------|----------|
| **Data ingested** | `data/processed/dataset_cleaned.parquet` (same) | Same |
| **Features used** | 13 (price/load lags + basic calendar) | 64 (all non-target columns) |
| **Target** | `P(T)` | `P(T)` |
| **Processing** | Select 13 cols, median-fill NaNs, temporal CV, 80/20 train/val | Select all 64 cols, median-fill, **hyperparameter tuning**, same CV and 80/20 |
| **Algorithm** | XGBoost (fixed config) | XGBoost (tuned config) |
| **Learns on** | Expanding past (80% of train period per fold) | Same, with full feature set and tuned params |
| **Validated on** | Last 20% of train period (monitoring only) | Same |
| **Tested against** | 7 × (3-month future block after 24h gap) | Same 7 test folds |
| **Metrics** | RMSE, MAE, MAPE, R², directional accuracy | Same + quantile metrics (P10/P50/P90) + baseline comparison |
| **Outputs** | `baseline_model.pkl`, CV results, diagnostics, plots | `enhanced_model.pkl`, `quantile_models.pkl`, CV results, comparison report/plots |

---

## 5. References in Code

- **Data load**: `train_baseline.py` (line ~41), `train_enhanced.py` (line ~47) — both use `dataset_cleaned.parquet`.
- **Baseline features**: `src/models/baseline.py` — `get_baseline_features()`.
- **Enhanced features**: `src/models/enhanced.py` — `get_enhanced_features()`.
- **Temporal CV**: `src/validation/temporal_cv.py` — `TemporalCV`, `temporal_cv_split()`.
- **Config**: `config/model_config.yaml` — `cv`, `baseline`, `enhanced`, `quantile`.
- **Metrics**: `src/validation/metrics.py` — `calculate_metrics`, `calculate_directional_accuracy`, etc.
- **Tuning**: `src/validation/hyperparameter_tuning.py` — `tune_with_baseline_comparison`, `get_param_distributions()`.
- **Dataset creation**: `regenerate_full_dataset.py`, `fix_data_issues.py`, `src/data_pipeline/merge_datasets.py`, `src/data_pipeline/feature_engineering.py`.
- **Data and column semantics**: `HISTORICAL_DATA_USAGE.md`.
