#!/usr/bin/env python3
"""
LSTM vs XGBoost comparison on same holdout and same test period.
Uses v2 data (dataset_cleaned.parquet), same holdout; LSTM uses contemporaneous features,
XGBoost uses lag features; both evaluated on the same test window.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# Optional backends
try:
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.preprocessing import MinMaxScaler


def get_v2_root():
    return Path(__file__).resolve().parent


def load_config(config_path=None):
    if config_path is None:
        config_path = get_v2_root() / "lstm_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_sequences(X, y, lookback):
    n = len(X)
    if n <= lookback:
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32), np.array([], dtype=np.float32)
    X_seq = np.zeros((n - lookback, lookback, X.shape[1]), dtype=np.float32)
    y_out = np.zeros(n - lookback, dtype=np.float32)
    for i in range(lookback, n):
        X_seq[i - lookback] = X[i - lookback : i]
        y_out[i - lookback] = y[i]
    return X_seq, y_out


def inverse_transform_price(scaled_1d, scaler, n_features, price_col_index=0):
    n = len(scaled_1d)
    dummy = np.zeros((n, n_features), dtype=np.float32)
    dummy[:, price_col_index] = np.asarray(scaled_1d).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_index]


def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def main():
    root = get_v2_root()
    config = load_config()
    dataset_path = root / config["data"]["dataset"]
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pd.read_parquet(dataset_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        df = df.sort_index()
    holdout_hours = config["data"].get("holdout_hours", 0) or 0
    if holdout_hours > 0:
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours)
        df = df[df.index <= holdout_start]
    xgb_mape = xgb_rmse = None
    print("=" * 60)
    print("LSTM vs XGBoost — same test period")
    print("=" * 60)
    print(f"Data: {len(df)} rows, {df.index.min()} to {df.index.max()}")

    target = config["data"]["target"]
    requested = config["data"].get("lstm_features") or []
    lstm_features = [f for f in requested if f in df.columns]
    if "P(T)" not in lstm_features:
        lstm_features = [target] + [f for f in lstm_features if f != target]
    else:
        lstm_features = ["P(T)"] + [f for f in lstm_features if f != "P(T)"]
    price_col_index = 0
    lookback = config["sequence"]["lookback_hours"]

    # Drop NaN
    valid = df[target].notna()
    for c in lstm_features:
        valid = valid & df[c].notna()
    df_clean = df.loc[valid].copy()
    X_raw = df_clean[lstm_features].values.astype(np.float32)
    y_raw = df_clean[target].values.astype(np.float32)
    n = len(X_raw)
    X_seq_raw, y_out = build_sequences(X_raw, y_raw, lookback)
    n_seq = len(y_out)

    train_frac = config["data"]["train_frac"]
    val_frac = config["data"]["val_frac"]
    test_frac = config["data"]["test_frac"]
    t1 = int(n_seq * train_frac)
    t2 = int(n_seq * (train_frac + val_frac))
    test_start_seq = t2
    test_end_seq = n_seq
    # Test period in original row index: rows [test_start_seq + lookback - 1, n_seq + lookback - 1] inclusive
    # i.e. sequence index test_start_seq predicts hour at original index test_start_seq + lookback
    test_start_row = test_start_seq + lookback
    test_end_row = n_seq + lookback  # last row index used for last sequence
    if test_end_row > n:
        test_end_row = n
    y_test_actual = y_raw[test_start_row:test_end_row]

    # ---------- LSTM ----------
    scaler = MinMaxScaler()
    X_train_flat = X_seq_raw[:t1].reshape(-1, X_seq_raw.shape[2])
    scaler.fit(X_train_flat)
    X_seq = np.zeros_like(X_seq_raw)
    for i in range(n_seq):
        X_seq[i] = scaler.transform(X_seq_raw[i])
    full_scaled = scaler.transform(X_raw)
    y_out_scaled = full_scaled[lookback:, price_col_index].astype(np.float32)
    X_test_lstm = X_seq[test_start_seq:test_end_seq]
    y_test_scaled = y_out_scaled[test_start_seq:test_end_seq]
    n_features = X_seq.shape[2]

    lstm_mape = lstm_rmse = None
    model_path = root / "results" / "lstm_model.keras"
    model_pt = root / "results" / "lstm_model.pt"
    if HAS_TF and model_path.exists():
        model = keras.models.load_model(model_path)
        y_pred_scaled = model.predict(X_test_lstm, verbose=0).flatten()
        y_pred_actual = inverse_transform_price(y_pred_scaled, scaler, n_features, price_col_index)
        # Align length if needed
        min_len = min(len(y_test_actual), len(y_pred_actual))
        lstm_mape = mape(y_test_actual[:min_len], y_pred_actual[:min_len])
        lstm_rmse = rmse(y_test_actual[:min_len], y_pred_actual[:min_len])
        print(f"\nLSTM (loaded {model_path.name}): MAPE={lstm_mape:.2f}%  RMSE={lstm_rmse:.2f}")
    elif HAS_TORCH and model_pt.exists():
        ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
        from train_lstm import build_model, LSTMModule
        model = build_model(config, n_features=ckpt["n_features"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(torch.from_numpy(X_test_lstm)).numpy().flatten()
        y_pred_actual = inverse_transform_price(y_pred_scaled, scaler, n_features, price_col_index)
        min_len = min(len(y_test_actual), len(y_pred_actual))
        lstm_mape = mape(y_test_actual[:min_len], y_pred_actual[:min_len])
        lstm_rmse = rmse(y_test_actual[:min_len], y_pred_actual[:min_len])
        print(f"\nLSTM (loaded {model_pt.name}): MAPE={lstm_mape:.2f}%  RMSE={lstm_rmse:.2f}")
    else:
        print("\nLSTM: No saved model found in v2/results/. Run: python v2/train_lstm.py")

    # ---------- XGBoost: same test period (test_start_row : test_end_row) ----------
    xgb_features = [
        "P(T-1)", "P(T-2)", "P(T-24)", "P(T-48)",
        "Hour", "DayOfWeek", "Month", "Season",
    ]
    xgb_features = [f for f in xgb_features if f in df_clean.columns]
    if not xgb_features:
        print("XGBoost: No lag features found in dataset.")
    elif HAS_XGB:
        # Train on data up to test start; test on same rows as LSTM
        X_train_xgb = df_clean.iloc[:test_start_row][xgb_features].fillna(0)
        y_train_xgb = df_clean.iloc[:test_start_row][target]
        X_test_xgb = df_clean.iloc[test_start_row:test_end_row][xgb_features].fillna(0)
        y_test_xgb = df_clean.iloc[test_start_row:test_end_row][target].values
        xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        xgb.fit(X_train_xgb, y_train_xgb)
        y_pred_xgb = xgb.predict(X_test_xgb)
        min_len = min(len(y_test_xgb), len(y_pred_xgb))
        xgb_mape = mape(y_test_xgb[:min_len], y_pred_xgb[:min_len])
        xgb_rmse = rmse(y_test_xgb[:min_len], y_pred_xgb[:min_len])
        print(f"XGBoost (lag features): MAPE={xgb_mape:.2f}%  RMSE={xgb_rmse:.2f}")
    elif not HAS_XGB:
        print("XGBoost: xgboost not installed.")

    print("\n" + "=" * 60)
    if lstm_mape is not None or xgb_mape is not None:
        print("Summary (same test period):")
        if lstm_mape is not None:
            print(f"  LSTM:    MAPE={lstm_mape:.2f}%  RMSE={lstm_rmse:.2f}")
        if xgb_mape is not None:
            print(f"  XGBoost: MAPE={xgb_mape:.2f}%  RMSE={xgb_rmse:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
