#!/usr/bin/env python3
"""
Bayesian LSTM tuning with nested time-series CV (5 folds + 48h gap).
Reduces overfitting to a single validation slice; each trial returns mean MAPE across folds.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import yaml

from train_lstm import (
    HAS_TF,
    HAS_TORCH,
    build_model,
    build_sequences,
    calculate_mape,
    get_v2_root,
    inverse_transform_price,
    load_config,
    load_data,
)
from tune_lstm_bayesian import _train_pytorch, _train_tensorflow

if HAS_TORCH:
    import torch
    import torch.nn as nn


def time_series_cv_score(
    X_raw,
    y_raw,
    n_features,
    price_col_index,
    trial_config,
    lookback,
    n_splits=5,
    gap_hours=48,
    trial=None,
):
    """
    Run time-series CV: 5 folds with gap between train and val.
    Returns (mean MAPE over folds, list of per-fold MAPEs).
    """
    X_seq_raw, y_out = build_sequences(X_raw, y_raw, lookback)
    n_seq = len(y_out)
    if n_seq < 500:
        return 1e6, []

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_hours)
    fold_mapes = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(np.arange(n_seq))):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        if len(val_idx) == 0:
            continue

        X_train_flat = X_seq_raw[train_idx].reshape(-1, X_seq_raw.shape[2])
        scaler = MinMaxScaler()
        scaler.fit(X_train_flat)
        X_seq = np.zeros_like(X_seq_raw)
        for i in range(n_seq):
            X_seq[i] = scaler.transform(X_seq_raw[i])
        full_scaled_arr = scaler.transform(X_raw)
        y_out_scaled = full_scaled_arr[lookback:, price_col_index].astype(np.float32)

        X_train = X_seq[train_idx]
        y_train = y_out_scaled[train_idx]
        X_val = X_seq[val_idx]
        y_val = y_out_scaled[val_idx]

        model = build_model(trial_config, n_features=X_seq_raw.shape[2])
        if HAS_TF:
            model = _train_tensorflow(model, X_train, y_train, X_val, y_val, trial_config)
            y_pred_scaled = model.predict(X_val, verbose=0).flatten()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _train_pytorch(model, X_train, y_train, X_val, y_val, trial_config, device)
            model.eval()
            with torch.no_grad():
                y_pred_scaled = (
                    model(torch.from_numpy(X_val).to(device)).cpu().numpy().flatten()
                )

        y_val_actual = inverse_transform_price(y_val, scaler, n_features, price_col_index)
        y_pred_actual = inverse_transform_price(
            y_pred_scaled, scaler, n_features, price_col_index
        )
        fold_mape = float(calculate_mape(y_val_actual, y_pred_actual))
        fold_mapes.append(fold_mape)

        mean_so_far = np.mean(fold_mapes)
        if trial is not None:
            trial.report(mean_so_far, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if not fold_mapes:
        return 1e6, []
    return float(np.mean(fold_mapes)), fold_mapes


def main():
    parser = argparse.ArgumentParser(
        description="LSTM Bayesian tuning with nested time-series CV (v2)"
    )
    parser.add_argument("--config", default=None, help="Path to lstm_config.yaml")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds")
    parser.add_argument("--gap-hours", type=int, default=48, help="Gap between train and val (samples)")
    args = parser.parse_args()

    opt = optuna

    if not HAS_TF and not HAS_TORCH:
        print("Install TensorFlow or PyTorch: pip install tensorflow | pip install torch")
        sys.exit(1)
    backend = "tensorflow" if HAS_TF else "pytorch"
    print(f"Backend: {backend}")
    print(f"CV: n_splits={args.n_splits}, gap_hours={args.gap_hours}")

    root = get_v2_root()
    config = load_config(args.config)
    price_col_index = 0
    target_col = config["data"]["target"]

    print("Loading data...")
    df, lstm_features, _ = load_data(config)
    X_df = df[lstm_features].copy()
    y_series = df[target_col].copy()
    valid = y_series.notna()
    for c in lstm_features:
        valid = valid & X_df[c].notna()
    X_df = X_df.loc[valid]
    y_series = y_series.loc[valid]
    X_raw = X_df.values.astype(np.float32)
    y_raw = y_series.values.astype(np.float32)
    n_features = X_raw.shape[1]
    print(f"  Rows: {len(X_raw)}, features: {n_features}")

    def objective(trial):
        # Focused search: simple arch, unidirectional, dropout ~0.20, lags
        lookback = trial.suggest_categorical("lookback", [24, 48])
        lstm_units_1 = trial.suggest_categorical("lstm_units_1", [24, 32, 48])
        lstm_units_2 = trial.suggest_categorical("lstm_units_2", [0, 16])  # 0 = single layer
        dropout = trial.suggest_float("dropout", 0.15, 0.25)
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        # bidirectional fixed False (simpler arch wins with lags)

        lstm_units = [lstm_units_1] if lstm_units_2 == 0 else [lstm_units_1, lstm_units_2]

        trial_config = copy.deepcopy(config)
        trial_config["sequence"]["lookback_hours"] = lookback
        trial_config["model"]["lstm_units"] = lstm_units
        trial_config["model"]["dropout"] = dropout
        trial_config["model"]["bidirectional"] = False
        trial_config["training"]["learning_rate"] = learning_rate
        trial_config["training"]["batch_size"] = batch_size

        mean_mape, fold_mapes = time_series_cv_score(
            X_raw,
            y_raw,
            n_features,
            price_col_index,
            trial_config,
            lookback,
            n_splits=args.n_splits,
            gap_hours=args.gap_hours,
            trial=trial,
        )
        trial.set_user_attr("fold_mapes", fold_mapes)
        return mean_mape

    print("\nStarting Bayesian optimization (nested CV + pruning)...")
    print("=" * 60)
    study = opt.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=2),
        study_name="lstm_nested_cv",
    )
    study.enqueue_trial(
        {
            "lookback": 24,
            "lstm_units_1": 32,
            "lstm_units_2": 0,
            "dropout": 0.20,
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nBest mean CV MAPE: {study.best_value:.2f}%")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_path = out_dir / "optuna_lstm_nested_cv_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"\nSaved: {trials_path}")

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        fold_mapes = t.user_attrs.get("fold_mapes", [])
        for fold, mape in enumerate(fold_mapes):
            row = {"trial_number": t.number, "fold": fold, "mape": mape}
            for k, v in t.params.items():
                row[k] = v
            rows.append(row)
    if rows:
        import pandas as pd
        folds_df = pd.DataFrame(rows)
        folds_path = out_dir / "optuna_lstm_nested_cv_folds.csv"
        folds_df.to_csv(folds_path, index=False)
        print(f"Saved: {folds_path}")

    best_params = study.best_params
    best_lstm_units = (
        [best_params["lstm_units_1"]]
        if best_params["lstm_units_2"] == 0
        else [best_params["lstm_units_1"], best_params["lstm_units_2"]]
    )
    best_config = {
        "sequence": {"lookback_hours": best_params["lookback"]},
        "model": {
            "lstm_units": best_lstm_units,
            "dropout": best_params["dropout"],
            "bidirectional": False,
        },
        "training": {
            "learning_rate": best_params["learning_rate"],
            "batch_size": best_params["batch_size"],
        },
    }
    best_path = out_dir / "optuna_lstm_nested_cv_best_params.yaml"
    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {best_path}")

    try:
        import optuna.importance as optuna_importance
        importances = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importance:")
        for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {param}: {imp:.3f}")
    except Exception:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
