#!/usr/bin/env python3
"""
LSTM Final Training: Train on 90% of non-holdout data, validate on 10%.
No test split - maximizes training data for final model before holdout inference.

Usage:
  python train_lstm_final.py --seed 123
"""

import argparse
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.preprocessing import MinMaxScaler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if HAS_TF:
        tf.random.set_seed(seed)


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "lstm_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_v2_root():
    return Path(__file__).resolve().parent


def load_data(config):
    """Load parquet, trim holdout, resolve lstm_features."""
    root = get_v2_root()
    dataset_name = config["data"]["dataset"]
    path = root / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or 'datetime' column")
    df = df.sort_index()
    holdout_hours = config["data"].get("holdout_hours", 0) or 0
    if holdout_hours > 0:
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours)
        df = df[df.index <= holdout_start]
    requested = config["data"].get("lstm_features")
    if not requested:
        requested = [c for c in df.columns if c != config["data"]["target"]]
    lstm_features = [f for f in requested if f in df.columns]
    if requested and len(lstm_features) < len(requested):
        missing = set(requested) - set(df.columns)
        print(f"Warning: requested features not in parquet (skipped): {sorted(missing)}", file=sys.stderr)
    if "P(T)" not in lstm_features:
        lstm_features = [config["data"]["target"]] + [f for f in lstm_features if f != config["data"]["target"]]
    else:
        lstm_features = ["P(T)"] + [f for f in lstm_features if f != "P(T)"]
    return df, lstm_features, config["data"]["target"]


def build_sequences(X, y, lookback):
    """Build sequences for LSTM."""
    n = len(X)
    if n <= lookback:
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32), np.array([], dtype=np.float32)
    X_seq = np.zeros((n - lookback, lookback, X.shape[1]), dtype=np.float32)
    y_out = np.zeros(n - lookback, dtype=np.float32)
    for i in range(lookback, n):
        X_seq[i - lookback] = X[i - lookback : i]
        y_out[i - lookback] = y[i]
    return X_seq, y_out


def inverse_transform_price(scaled_price_1d, scaler, n_features, price_col_index=0):
    """Inverse transform price only."""
    n = len(scaled_price_1d)
    dummy = np.zeros((n, n_features), dtype=np.float32)
    dummy[:, price_col_index] = np.asarray(scaled_price_1d).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_index]


class LSTMModule(nn.Module):
    def __init__(self, n_features, lstm_units, dropout=0.2, dense_units=16, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        layers_list = []
        for i, u in enumerate(lstm_units):
            in_size = n_features if i == 0 else lstm_units[i - 1] * (2 if bidirectional else 1)
            layers_list.append(
                nn.LSTM(in_size, u, batch_first=True, dropout=0, bidirectional=bidirectional)
            )
        self.lstms = nn.ModuleList(layers_list)
        self.dropout = nn.Dropout(dropout)
        last_dim = lstm_units[-1] * (2 if bidirectional else 1)
        self.dense = nn.Sequential(
            nn.Linear(last_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x):
        h = x
        for i, lstm in enumerate(self.lstms):
            h, _ = lstm(h)
            h = self.dropout(h)
        h = h[:, -1, :]
        return self.dense(h).squeeze(-1)


def build_model(config, n_features):
    lstm_units = config["model"]["lstm_units"]
    dropout = config["model"].get("dropout", 0.2)
    dense_units = config["model"].get("dense_units", 16)
    lookback = config["sequence"]["lookback_hours"]
    bidirectional = config["model"].get("bidirectional", False)

    if HAS_TF:
        inputs = keras.Input(shape=(lookback, n_features))
        x = inputs
        for i, u in enumerate(lstm_units):
            return_seq = i < len(lstm_units) - 1
            if bidirectional:
                x = layers.Bidirectional(layers.LSTM(u, return_sequences=return_seq))(x)
            else:
                x = layers.LSTM(u, return_sequences=return_seq)(x)
            x = layers.Dropout(dropout)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        return keras.Model(inputs, outputs)
    if HAS_TORCH:
        return LSTMModule(
            n_features, lstm_units, dropout=dropout, dense_units=dense_units, bidirectional=bidirectional
        )
    raise RuntimeError("Install TensorFlow or PyTorch")


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
    parser = argparse.ArgumentParser(description="Final LSTM training (90/10 split, no test)")
    parser.add_argument("--config", default=None, help="Path to lstm_config.yaml")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs (overrides config)")
    args = parser.parse_args()

    if not HAS_TF and not HAS_TORCH:
        print("Install TensorFlow or PyTorch")
        sys.exit(1)
    backend = "tensorflow" if HAS_TF else "pytorch"
    print(f"Backend: {backend}")

    config = load_config(args.config)
    set_seed(args.seed)
    root = get_v2_root()
    target_col = config["data"]["target"]
    price_col_index = 0

    print("=" * 60)
    print("LSTM FINAL TRAINING — 90/10 split, no test set")
    print("=" * 60)

    print("\n1. Loading data (excluding holdout)...")
    df, lstm_features, _ = load_data(config)
    print(f"   Rows: {len(df)}, LSTM features: {len(lstm_features)} (P(T) first)")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    X_df = df[lstm_features].copy()
    y_series = df[target_col].copy()
    valid = y_series.notna()
    for c in lstm_features:
        valid = valid & X_df[c].notna()
    X_df = X_df.loc[valid]
    y_series = y_series.loc[valid]
    print(f"   After dropping NaN: {len(X_df)} rows")

    lookback = config["sequence"]["lookback_hours"]
    X_raw = X_df.values.astype(np.float32)
    y_raw = y_series.values.astype(np.float32)

    print(f"\n2. Building sequences (lookback={lookback})...")
    X_seq_raw, y_out = build_sequences(X_raw, y_raw, lookback)
    n_seq = len(y_out)

    # FINAL TRAINING: 90% train, 10% val, NO test
    train_frac = 0.9
    val_frac = 0.1
    t1 = int(n_seq * train_frac)
    train_idx = slice(0, t1)
    val_idx = slice(t1, n_seq)

    # MinMaxScaler fit on train only
    X_train_flat = X_seq_raw[train_idx].reshape(-1, X_seq_raw.shape[2])
    scaler = MinMaxScaler()
    scaler.fit(X_train_flat)
    X_seq = np.zeros_like(X_seq_raw)
    for i in range(n_seq):
        X_seq[i] = scaler.transform(X_seq_raw[i])
    full_scaled = scaler.transform(X_raw)
    y_out_scaled = full_scaled[lookback:, price_col_index].astype(np.float32)
    print(f"   Sequences: {X_seq.shape[0]}, shape (samples, lookback, features) = {X_seq.shape}")

    X_train = X_seq[train_idx]
    y_train = y_out_scaled[train_idx]
    X_val = X_seq[val_idx]
    y_val = y_out_scaled[val_idx]
    print(f"\n3. Split: train={len(y_train)}, val={len(y_val)} (NO test — all data for final model)")

    n_features = X_seq.shape[2]
    model = build_model(config, n_features=n_features)
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]

    if HAS_TF:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"],
        )
        callbacks = []
        if not args.no_early_stop:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config["training"]["early_stopping_patience"],
                    restore_best_weights=True,
                )
            )
        print("\n4. Training (TensorFlow)...")
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        y_pred_scaled = model.predict(X_val, verbose=0).flatten()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        if config["training"].get("reduce_lr_patience"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=config["training"].get("reduce_lr_factor", 0.5),
                patience=config["training"]["reduce_lr_patience"],
            )
        patience = config["training"]["early_stopping_patience"] if not args.no_early_stop else None
        best_val_loss = float("inf")
        best_state = None
        wait = 0
        print("\n4. Training (PyTorch)...")
        for epoch in range(epochs):
            model.train()
            perm = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                idx = perm[start : start + batch_size]
                xb = torch.from_numpy(X_train[idx]).to(device)
                yb = torch.from_numpy(y_train[idx]).to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).to(device)
                yv = torch.from_numpy(y_val).to(device)
                val_pred = model(xv)
                val_loss = nn.functional.mse_loss(val_pred, yv).item()
            if scheduler:
                scheduler.step(val_loss)
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if best_state is not None:
                            model.load_state_dict(best_state)
                        break
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}")
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(torch.from_numpy(X_val).to(device)).cpu().numpy().flatten()

    # Validation metrics (no test set in final training)
    y_pred_actual = inverse_transform_price(y_pred_scaled, scaler, n_features, price_col_index)
    y_val_actual = inverse_transform_price(y_val, scaler, n_features, price_col_index)
    mape = calculate_mape(y_val_actual, y_pred_actual)
    rmse = calculate_rmse(y_val_actual, y_pred_actual)
    print("\n5. Validation set metrics (inverse-transformed):")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   RMSE: {rmse:.2f}")

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save final model with "_final" suffix
    if HAS_TF:
        model_path = out_dir / "lstm_model_final.keras"
        model.save(model_path)
    else:
        model_path = out_dir / "lstm_model_final.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config, "n_features": n_features},
            model_path,
        )
    print(f"\n6. Model saved: {model_path}")

    scaler_path = out_dir / "lstm_scaler_final.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "feature_cols": lstm_features,
            "lookback": lookback,
            "price_col_index": price_col_index,
        },
        scaler_path,
    )
    print(f"   Scaler/metadata saved: {scaler_path}")

    metrics = {"val_mape": float(mape), "val_rmse": float(rmse), "seed": args.seed}
    metrics_path = out_dir / "lstm_metrics_final.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"   Metrics saved: {metrics_path}")
    print("\nDone. Model ready for holdout inference.")


if __name__ == "__main__":
    main()
