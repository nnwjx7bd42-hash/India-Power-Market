# Auxiliary scripts

Non-essential scripts for comparison, tuning, auditing, and diagnostics. Run all commands from the **project root** (v1).

## Scripts

| Script | Purpose |
|--------|--------|
| `compare_lstm_xgb.py` | Compare LSTM vs XGB holdout performance and plot |
| `compare_tuning_results.py` | Compare Optuna tuning runs (CSV/YAML under v2/results) |
| `inspect_test_period.py` | Inspect test-period metrics and plots |
| `plot_holdout.py` | Plot holdout actuals vs predictions from v2 results |
| `run_multi_seed.py` | Run LSTM training with multiple seeds |
| `apply_best_params.py` | Apply best Optuna params to lstm_config.yaml |
| `audit_full_pipeline.py` | Full pipeline re-run audit (all layers, compares to saved metrics) |
| `stress_test_pipeline.py` | Pipeline stress tests (pytest) — data, point forecast, probabilistic, scenarios, backtest |
| `model_comparison.py` | Baseline vs enhanced model comparison utilities |
| `diagnostics.py` | Residual diagnostics and plots |
| `tune_quantile_xgb.py` | Optuna tuning for v4 quantile XGBoost |

## Run from project root

```bash
# Audit (full pipeline re-run, ~7+ min)
python auxiliary/audit_full_pipeline.py

# Stress tests (pytest)
pytest auxiliary/stress_test_pipeline.py -v --tb=short

# Examples
python auxiliary/plot_holdout.py
python auxiliary/compare_lstm_xgb.py
python auxiliary/tune_quantile_xgb.py --trials 20
```

Results (e.g. audit JSON, comparison plots) are written under `v2/results`, `v4/results`, or `auxiliary/` as noted in each script.
