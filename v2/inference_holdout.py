#!/usr/bin/env python3
"""
Holdout Inference: Run both LSTM and XGBoost final models on the held-out 168 hours.
Compares predictions and reports MAPE/RMSE for both models.

Usage:
  python inference_holdout.py
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.preprocessing import MinMaxScaler


def get_v2_root():
    return Path(__file__).resolve().parent


def load_config(config_path=None):
    if config_path is None:
        config_path = get_v2_root() / "lstm_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_sequences(X, lookback):
    """Build sequences for prediction (no y needed)."""
    n = len(X)
    if n <= lookback:
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32)
    X_seq = np.zeros((n - lookback, lookback, X.shape[1]), dtype=np.float32)
    for i in range(lookback, n):
        X_seq[i - lookback] = X[i - lookback : i]
    return X_seq


def inverse_transform_price(scaled_price_1d, scaler, n_features, price_col_index=0):
    """Inverse transform price only."""
    n = len(scaled_price_1d)
    dummy = np.zeros((n, n_features), dtype=np.float32)
    dummy[:, price_col_index] = np.asarray(scaled_price_1d).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_index]


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


# Import LSTMModule from train_lstm_final (or define here)
class LSTMModule(torch.nn.Module):
    def __init__(self, n_features, lstm_units, dropout=0.2, dense_units=16, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        layers_list = []
        for i, u in enumerate(lstm_units):
            in_size = n_features if i == 0 else lstm_units[i - 1] * (2 if bidirectional else 1)
            layers_list.append(
                torch.nn.LSTM(in_size, u, batch_first=True, dropout=0, bidirectional=bidirectional)
            )
        self.lstms = torch.nn.ModuleList(layers_list)
        self.dropout = torch.nn.Dropout(dropout)
        last_dim = lstm_units[-1] * (2 if bidirectional else 1)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(last_dim, dense_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dense_units, 1),
        )

    def forward(self, x):
        h = x
        for lstm in self.lstms:
            h, _ = lstm(h)
            h = self.dropout(h)
        h = h[:, -1, :]
        return self.dense(h).squeeze(-1)


def build_lstm_model(config, n_features):
    """Build LSTM model from config."""
    lstm_units = config["model"]["lstm_units"]
    dropout = config["model"].get("dropout", 0.2)
    dense_units = config["model"].get("dense_units", 16)
    bidirectional = config["model"].get("bidirectional", False)
    return LSTMModule(n_features, lstm_units, dropout=dropout, dense_units=dense_units, bidirectional=bidirectional)


def main():
    parser = argparse.ArgumentParser(description="Run inference on holdout period")
    parser.add_argument("--config", default=None, help="Path to lstm_config.yaml")
    args = parser.parse_args()

    root = get_v2_root()
    config = load_config(args.config)
    target_col = config["data"]["target"]
    holdout_hours = config["data"].get("holdout_hours", 168)

    print("=" * 60)
    print(f"HOLDOUT INFERENCE ({holdout_hours} hours)")
    print("=" * 60)

    # Load full dataset (including holdout)
    print("\n1. Loading full dataset...")
    dataset_path = root / config["data"]["dataset"]
    df_full = pd.read_parquet(dataset_path)
    if not isinstance(df_full.index, pd.DatetimeIndex):
        if "datetime" in df_full.columns:
            df_full = df_full.set_index("datetime")
    df_full = df_full.sort_index()
    print(f"   Total rows: {len(df_full)}")
    print(f"   Date range: {df_full.index.min()} to {df_full.index.max()}")

    # Split into non-holdout and holdout
    holdout_start = df_full.index.max() - pd.Timedelta(hours=holdout_hours - 1)
    df_train = df_full[df_full.index < holdout_start]
    df_holdout = df_full[df_full.index >= holdout_start]
    print(f"   Non-holdout: {len(df_train)} rows (up to {df_train.index.max()})")
    print(f"   Holdout: {len(df_holdout)} rows ({df_holdout.index.min()} to {df_holdout.index.max()})")

    # Actual prices in holdout
    y_holdout_actual = df_holdout[target_col].values

    # ==================== LSTM INFERENCE ====================
    print("\n2. LSTM Inference...")

    # Load LSTM model and scaler
    lstm_model_path = root / "results" / "lstm_model_final.pt"
    lstm_scaler_path = root / "results" / "lstm_scaler_final.joblib"

    if not lstm_model_path.exists():
        print(f"   ERROR: LSTM model not found: {lstm_model_path}")
        print("   Run: python train_lstm_final.py --seed 123")
        lstm_pred = None
    else:
        if not HAS_TORCH:
            print("   ERROR: PyTorch required for LSTM inference")
            lstm_pred = None
        else:
            # Load scaler and model
            scaler_data = joblib.load(lstm_scaler_path)
            scaler = scaler_data["scaler"]
            lstm_features = scaler_data["feature_cols"]
            lookback = scaler_data["lookback"]
            price_col_index = scaler_data["price_col_index"]

            ckpt = torch.load(lstm_model_path, map_location="cpu", weights_only=False)
            n_features = ckpt["n_features"]
            model_config = ckpt["config"]

            model = build_lstm_model(model_config, n_features)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # Need lookback hours before holdout + holdout itself for sequences
            # Get data from (holdout_start - lookback) to end
            seq_start = holdout_start - pd.Timedelta(hours=lookback)
            df_for_seq = df_full[df_full.index >= seq_start][lstm_features].copy()

            # Drop NaN
            valid = df_for_seq.notna().all(axis=1)
            df_for_seq = df_for_seq.loc[valid]

            X_raw = df_for_seq.values.astype(np.float32)

            # Scale using final scaler
            X_scaled = scaler.transform(X_raw)

            # Build sequences
            X_seq = build_sequences(X_scaled, lookback)

            # The sequences predict for indices [lookback, lookback+1, ..., len-1]
            # We need predictions for holdout timestamps
            # Sequence i predicts for row (lookback + i) in df_for_seq
            # We want predictions for holdout rows

            # Get indices in df_for_seq that are holdout
            holdout_mask = df_for_seq.index >= holdout_start
            holdout_indices_in_df = np.where(holdout_mask)[0]

            # Sequence index = row_index - lookback
            holdout_seq_indices = holdout_indices_in_df - lookback
            holdout_seq_indices = holdout_seq_indices[holdout_seq_indices >= 0]
            holdout_seq_indices = holdout_seq_indices[holdout_seq_indices < len(X_seq)]

            X_holdout = X_seq[holdout_seq_indices]

            with torch.no_grad():
                y_pred_scaled = model(torch.from_numpy(X_holdout)).numpy().flatten()

            lstm_pred = inverse_transform_price(y_pred_scaled, scaler, n_features, price_col_index)
            print(f"   LSTM predictions: {len(lstm_pred)} samples")

    # ==================== XGBOOST INFERENCE ====================
    print("\n3. XGBoost Inference...")

    xgb_model_path = root / "results" / "xgb_model_final.pkl"

    if not xgb_model_path.exists():
        print(f"   ERROR: XGBoost model not found: {xgb_model_path}")
        print("   Run: python train_xgb_final.py")
        xgb_pred = None
    else:
        model_data = joblib.load(xgb_model_path)
        xgb_model = model_data["model"]
        xgb_features = model_data["feature_names"]

        # Prepare holdout features
        X_holdout_xgb = df_holdout[xgb_features].copy()

        # Fill missing
        for f in xgb_features:
            if f not in X_holdout_xgb.columns:
                X_holdout_xgb[f] = 0
        X_holdout_xgb = X_holdout_xgb.fillna(0)

        xgb_pred = xgb_model.predict(X_holdout_xgb[xgb_features].values)
        print(f"   XGBoost predictions: {len(xgb_pred)} samples")

    # ==================== COMPARE RESULTS ====================
    print("\n" + "=" * 60)
    print("HOLDOUT INFERENCE RESULTS")
    print("=" * 60)

    results = {}

    # Align lengths
    min_len = len(y_holdout_actual)
    if lstm_pred is not None:
        min_len = min(min_len, len(lstm_pred))
    if xgb_pred is not None:
        min_len = min(min_len, len(xgb_pred))

    y_actual = y_holdout_actual[:min_len]
    holdout_timestamps = df_holdout.index[:min_len]

    print(f"\nHoldout period: {holdout_timestamps.min()} to {holdout_timestamps.max()}")
    print(f"Samples: {min_len}")

    print(f"\n{'Model':<20} {'MAPE %':>10} {'RMSE':>12}")
    print("-" * 45)

    if lstm_pred is not None:
        lstm_pred = lstm_pred[:min_len]
        lstm_mape = calculate_mape(y_actual, lstm_pred)
        lstm_rmse = calculate_rmse(y_actual, lstm_pred)
        print(f"{'LSTM (final)':<20} {lstm_mape:>10.2f} {lstm_rmse:>12.2f}")
        results["lstm_mape"] = float(lstm_mape)
        results["lstm_rmse"] = float(lstm_rmse)
    else:
        lstm_mape = None

    if xgb_pred is not None:
        xgb_pred = xgb_pred[:min_len]
        xgb_mape = calculate_mape(y_actual, xgb_pred)
        xgb_rmse = calculate_rmse(y_actual, xgb_pred)
        print(f"{'XGBoost (final)':<20} {xgb_mape:>10.2f} {xgb_rmse:>12.2f}")
        results["xgb_mape"] = float(xgb_mape)
        results["xgb_rmse"] = float(xgb_rmse)
    else:
        xgb_mape = None

    print("-" * 45)

    # Determine winner
    if lstm_mape is not None and xgb_mape is not None:
        if lstm_mape < xgb_mape:
            print(f"\nBest model: LSTM ({lstm_mape:.2f}% vs {xgb_mape:.2f}%)")
            results["best_model"] = "LSTM"
        else:
            print(f"\nBest model: XGBoost ({xgb_mape:.2f}% vs {lstm_mape:.2f}%)")
            results["best_model"] = "XGBoost"

    print("=" * 60)

    # Save predictions
    out_dir = root / "results"
    predictions_df = pd.DataFrame({
        "timestamp": holdout_timestamps,
        "actual": y_actual,
    })
    if lstm_pred is not None:
        predictions_df["lstm_pred"] = lstm_pred
    if xgb_pred is not None:
        predictions_df["xgb_pred"] = xgb_pred

    predictions_path = out_dir / "holdout_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved: {predictions_path}")

    # Save summary
    results["holdout_hours"] = min_len
    results["holdout_start"] = str(holdout_timestamps.min())
    results["holdout_end"] = str(holdout_timestamps.max())
    summary_path = out_dir / "holdout_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    main()
