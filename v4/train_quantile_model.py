#!/usr/bin/env python3
"""
V4 — Train the multi-quantile XGBoost planning model.

End-to-end: load planning dataset, split (train / val / test / holdout),
train with config or tuned params, evaluate all probabilistic metrics,
save model + metrics.

Usage:
  python train_quantile_model.py [--config config/planning_config.yaml]
                                 [--tuned-params results/optuna_quantile_best_params.yaml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

V4_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V4_ROOT))

from models.quantile_xgb import QuantileForecaster


def load_config(path=None):
    if path is None:
        path = V4_ROOT / "config" / "planning_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_tuned_params(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train v4 quantile XGBoost")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tuned-params", default=None,
                        help="YAML from Optuna (overrides model section in config)")
    args = parser.parse_args()

    config = load_config(args.config)
    tuned = load_tuned_params(args.tuned_params)

    print("=" * 60)
    print("V4 — QUANTILE XGB TRAINING")
    print("=" * 60)

    # ----------------------------------------------------------------- load
    dataset_path = V4_ROOT / "data" / "planning_dataset.parquet"
    df = pd.read_parquet(dataset_path)
    target = config["data"]["target"]

    feat_cfg = config["features"]
    feature_names = (
        feat_cfg["calendar"] + feat_cfg["weather"] + feat_cfg["load"]
        + feat_cfg["anchors"] + feat_cfg["interactions"]
    )
    feature_names = [f for f in feature_names if f in df.columns]

    print(f"\nDataset: {len(df):,} rows, {len(feature_names)} features")
    print(f"Date range: {df.index.min()} — {df.index.max()}")

    # ----------------------------------------------------------- split
    tz = df.index.tz
    train_end = pd.Timestamp(config["data"]["train_end"], tz=tz)
    val_end = pd.Timestamp(config["data"]["val_end"], tz=tz)
    holdout_hours = config["data"].get("holdout_hours", 168)
    holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours - 1)

    df_train = df[df.index <= train_end]
    df_val = df[(df.index > train_end) & (df.index <= val_end)]
    df_test = df[(df.index > val_end) & (df.index < holdout_start)]
    df_holdout = df[df.index >= holdout_start]

    print(f"\nSplit:")
    print(f"  Train:   {len(df_train):>7,}  ({df_train.index.min()} → {df_train.index.max()})")
    print(f"  Val:     {len(df_val):>7,}  ({df_val.index.min()} → {df_val.index.max()})")
    print(f"  Test:    {len(df_test):>7,}  ({df_test.index.min()} → {df_test.index.max()})")
    print(f"  Holdout: {len(df_holdout):>7,}  ({df_holdout.index.min()} → {df_holdout.index.max()})")

    X_train = df_train[feature_names].values
    y_train = df_train[target].values
    X_val = df_val[feature_names].values
    y_val = df_val[target].values
    X_test = df_test[feature_names].values
    y_test = df_test[target].values
    X_hold = df_holdout[feature_names].values
    y_hold = df_holdout[target].values

    # ----------------------------------------------------------- model params
    quantiles = config["quantiles"]
    mcfg = config["model"]

    # Override with tuned params if available
    if tuned:
        print("\nUsing Optuna-tuned params:")
        model_kwargs = dict(
            quantiles=quantiles,
            max_depth=tuned.get("max_depth", mcfg["max_depth"]),
            learning_rate=tuned.get("lr", mcfg["learning_rate"]),
            n_estimators=mcfg["n_estimators"],
            min_child_weight=tuned.get("mcw", mcfg.get("min_child_weight", 50)),
            subsample=tuned.get("subsample", mcfg["subsample"]),
            colsample_bytree=tuned.get("colsample", mcfg["colsample_bytree"]),
            reg_alpha=tuned.get("reg_alpha", mcfg.get("reg_alpha", 0.1)),
            reg_lambda=tuned.get("reg_lambda", mcfg.get("reg_lambda", 1.0)),
            early_stopping_rounds=mcfg["early_stopping_rounds"],
            random_state=config.get("random_state", 42),
        )
    else:
        print("\nUsing default config params:")
        model_kwargs = dict(
            quantiles=quantiles,
            max_depth=mcfg["max_depth"],
            learning_rate=mcfg["learning_rate"],
            n_estimators=mcfg["n_estimators"],
            min_child_weight=mcfg.get("min_child_weight", 50),
            subsample=mcfg["subsample"],
            colsample_bytree=mcfg["colsample_bytree"],
            reg_alpha=mcfg.get("reg_alpha", 0.1),
            reg_lambda=mcfg.get("reg_lambda", 1.0),
            early_stopping_rounds=mcfg["early_stopping_rounds"],
            random_state=config.get("random_state", 42),
        )

    for k, v in model_kwargs.items():
        if k != "quantiles":
            print(f"  {k}: {v}")

    # ----------------------------------------------------------- train
    print("\nTraining ...")
    model = QuantileForecaster(**model_kwargs)
    val_metrics = model.train(X_train, y_train, X_val, y_val,
                              feature_names=feature_names)

    print(f"\n--- Validation metrics ---")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ----------------------------------------------------------- test
    print("\n--- Test metrics ---")
    q_test = model.predict(X_test, feature_names=feature_names)
    test_metrics = model.evaluate(y_test, q_test)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ----------------------------------------------------------- holdout
    print("\n--- Holdout metrics ---")
    q_hold = model.predict(X_hold, feature_names=feature_names)
    hold_metrics = model.evaluate(y_hold, q_hold)
    for k, v in hold_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ----------------------------------------------------------- save
    out_dir = V4_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    print(f"\nModel saved: {out_dir / 'quantile_model.json'}")

    # Save quantile forecasts for holdout
    q_cols = [f"q{int(q*100):02d}" for q in quantiles]
    holdout_df = pd.DataFrame(q_hold, columns=q_cols, index=df_holdout.index)
    holdout_df.insert(0, "actual", y_hold)
    holdout_df.to_csv(out_dir / "holdout_quantile_forecasts.csv")
    print(f"Holdout quantile forecasts: {out_dir / 'holdout_quantile_forecasts.csv'}")

    # Save all metrics
    all_metrics = {
        "validation": {k: float(v) for k, v in val_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
        "holdout": {k: float(v) for k, v in hold_metrics.items()},
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "quantiles": [float(q) for q in quantiles],
    }
    metrics_path = out_dir / "quantile_model_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f, default_flow_style=False, sort_keys=False)
    print(f"Metrics: {metrics_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
