#!/usr/bin/env python3
"""
Bayesian hyperparameter tuning for the v4 multi-quantile XGBoost model.

Minimises average pinball loss on validation across all quantiles.

Usage:
  python tune_quantile_xgb.py [--config ../config/planning_config.yaml] [--trials 80]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT))

from evaluation.pinball_loss import avg_pinball_loss


def load_config(config_path=None):
    if config_path is None:
        config_path = V4_ROOT / "config" / "planning_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_and_split(config):
    """Load planning dataset and split into train / val."""
    dataset_path = V4_ROOT / "data" / "planning_dataset.parquet"
    df = pd.read_parquet(dataset_path)
    target = config["data"]["target"]

    train_end = pd.Timestamp(config["data"]["train_end"], tz=df.index.tz)
    val_end = pd.Timestamp(config["data"]["val_end"], tz=df.index.tz)

    feat_cfg = config["features"]
    feature_names = (
        feat_cfg["calendar"] + feat_cfg["weather"] + feat_cfg["load"]
        + feat_cfg["anchors"] + feat_cfg["interactions"]
    )
    feature_names = [f for f in feature_names if f in df.columns]

    df_train = df[df.index <= train_end]
    df_val = df[(df.index > train_end) & (df.index <= val_end)]

    X_train = df_train[feature_names].values
    y_train = df_train[target].values
    X_val = df_val[feature_names].values
    y_val = df_val[target].values

    return X_train, y_train, X_val, y_val, feature_names


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for quantile XGBoost")
    parser.add_argument("--config", default=None)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    n_trials = args.trials or config.get("tuning", {}).get("n_trials", 80)
    quantiles = np.array(config["quantiles"])
    seed = config.get("random_state", 42)

    print("=" * 60)
    print("V4 — QUANTILE XGB HYPERPARAMETER TUNING")
    print("=" * 60)

    X_train, y_train, X_val, y_val, feature_names = load_and_split(config)
    print(f"Train: {len(X_train):,}   Val: {len(X_val):,}")
    print(f"Features: {len(feature_names)}   Quantiles: {len(quantiles)}")

    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain, feature_names=feature_names)

    def objective(trial):
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": quantiles,
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.10, log=True),
            "min_child_weight": trial.suggest_int("mcw", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "seed": seed,
        }
        booster = xgb.train(
            params, dtrain,
            num_boost_round=500,
            early_stopping_rounds=20,
            evals=[(dval, "Val")],
            verbose_eval=False,
        )
        preds = booster.predict(dval)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        preds = np.sort(preds, axis=1)  # monotonicity
        return avg_pinball_loss(y_val, preds, quantiles)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest avg pinball: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best params
    out_dir = V4_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_params.copy()
    best["best_avg_pinball"] = float(study.best_value)

    out_path = out_dir / "optuna_quantile_best_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(best, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved: {out_path}")

    # Save trials history
    trials_df = study.trials_dataframe()
    trials_path = out_dir / "optuna_quantile_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"Trials: {trials_path}")


if __name__ == "__main__":
    main()
