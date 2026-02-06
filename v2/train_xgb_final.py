#!/usr/bin/env python3
"""
XGBoost Final Training: Train on 90% of non-holdout data, validate on 10%.
Uses curated 19 features for fair comparison with LSTM.

Usage:
  python train_xgb_final.py
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_v2_root():
    return Path(__file__).resolve().parent


def load_config(config_path=None):
    if config_path is None:
        config_path = get_v2_root() / "lstm_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_curated_19_features():
    """Return curated 19-feature set."""
    return [
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)', 'P_T-168', 'Price_MA_24h',
        'Hour', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsMonday', 'IsWeekend', 'Month', 'Season',
        'CDH',
        'Net_Load', 'Solar_Ramp',
        'Load_MA_24h',
        'Hour_x_CDH',
    ]


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def main():
    parser = argparse.ArgumentParser(description="Final XGBoost training (90/10 split)")
    parser.add_argument("--config", default=None, help="Path to lstm_config.yaml (for holdout settings)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if not HAS_XGB:
        print("Install xgboost: pip install xgboost")
        sys.exit(1)

    config = load_config(args.config)
    root = get_v2_root()
    target_col = config["data"]["target"]

    print("=" * 60)
    print("XGBOOST FINAL TRAINING — 90/10 split, 19 curated features")
    print("=" * 60)

    # Load data
    print("\n1. Loading data (excluding holdout)...")
    dataset_path = root / config["data"]["dataset"]
    df = pd.read_parquet(dataset_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
    df = df.sort_index()

    # Exclude holdout
    holdout_hours = config["data"].get("holdout_hours", 0) or 0
    if holdout_hours > 0:
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours)
        df = df[df.index <= holdout_start]

    print(f"   Rows: {len(df)}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Get curated features
    print("\n2. Selecting features...")
    all_features = get_curated_19_features()
    features = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"   Warning: Missing features (skipped): {missing}")
    print(f"   Using {len(features)} features")

    # Prepare data
    X = df[features].copy()
    y = df[target_col].copy()

    # Drop NaN
    valid = y.notna()
    for c in features:
        valid = valid & X[c].notna()
    X = X.loc[valid]
    y = y.loc[valid]
    print(f"   After dropping NaN: {len(X)} rows")

    # 90/10 train/val split
    n = len(X)
    t1 = int(n * 0.9)
    X_train, X_val = X.iloc[:t1], X.iloc[t1:]
    y_train, y_val = y.iloc[:t1], y.iloc[t1:]
    print(f"\n3. Split: train={len(X_train)}, val={len(X_val)} (NO test — all data for final model)")

    # Train XGBoost
    print("\n4. Training XGBoost...")
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print(f"   Best iteration: {model.best_iteration}")

    # Validation metrics
    y_pred = model.predict(X_val)
    mape = calculate_mape(y_val.values, y_pred)
    rmse = calculate_rmse(y_val.values, y_pred)
    print("\n5. Validation set metrics:")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   RMSE: {rmse:.2f}")

    # Save model
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_model_final.pkl"
    model_data = {
        "model": model,
        "feature_names": features,
        "training_date": pd.Timestamp.now().isoformat(),
    }
    joblib.dump(model_data, model_path)
    print(f"\n6. Model saved: {model_path}")

    metrics = {"val_mape": float(mape), "val_rmse": float(rmse), "seed": args.seed, "n_features": len(features)}
    metrics_path = out_dir / "xgb_metrics_final.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"   Metrics saved: {metrics_path}")
    print("\nDone. Model ready for holdout inference.")


if __name__ == "__main__":
    main()
