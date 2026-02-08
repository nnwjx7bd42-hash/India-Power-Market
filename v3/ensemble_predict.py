#!/usr/bin/env python3
"""
XGBoost + LSTM Ensemble Prediction

Load trained XGBoost and LSTM models, generate predictions on the same test period,
combine with weighted average (with optional validation-set weight tuning).

Usage:
  python ensemble_predict.py [--config ensemble_config.yaml] [--tune-weights] [--out results/]
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# Add v2 to path for importing train_lstm utilities
PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_ROOT = PROJECT_ROOT / "v3"
V2_ROOT = PROJECT_ROOT / "v2"
sys.path.insert(0, str(V2_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.preprocessing import MinMaxScaler

# Import from v2/train_lstm.py
from train_lstm import build_model, build_sequences


def load_config(config_path=None):
    """Load ensemble configuration."""
    if config_path is None:
        config_path = V3_ROOT / "ensemble_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f), Path(config_path).parent


def resolve_path(path_str, base_dir):
    """Resolve relative path from base directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_data(config, base_dir):
    """Load and prepare data (same logic as v2/train_lstm.py)."""
    dataset_path = resolve_path(config["data"]["dataset"], base_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
    df = df.sort_index()

    # Apply holdout
    holdout_hours = config["data"].get("holdout_hours", 0) or 0
    if holdout_hours > 0:
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours)
        df = df[df.index <= holdout_start]

    return df


def prepare_lstm_data(df, config):
    """Prepare LSTM sequences and split (same as v2/train_lstm.py)."""
    target_col = config["data"]["target"]
    lstm_features = config.get("lstm_features", [])
    if not lstm_features:
        lstm_features = [c for c in df.columns if c != target_col]

    # Ensure P(T) first
    if "P(T)" not in lstm_features:
        lstm_features = [target_col] + [f for f in lstm_features if f != target_col]
    else:
        lstm_features = ["P(T)"] + [f for f in lstm_features if f != "P(T)"]

    # Filter to existing columns
    lstm_features = [f for f in lstm_features if f in df.columns]

    # Drop NaN
    X_df = df[lstm_features].copy()
    y_series = df[target_col].copy()
    valid = y_series.notna()
    for c in lstm_features:
        valid = valid & X_df[c].notna()
    X_df = X_df.loc[valid]
    y_series = y_series.loc[valid]

    lookback = config["sequence"]["lookback_hours"]
    X_raw = X_df.values.astype(np.float32)
    y_raw = y_series.values.astype(np.float32)

    # Build sequences
    X_seq_raw, y_out = build_sequences(X_raw, y_raw, lookback)
    n_seq = len(y_out)

    # Split
    train_frac = config["data"]["train_frac"]
    val_frac = config["data"]["val_frac"]
    t1 = int(n_seq * train_frac)
    t2 = int(n_seq * (train_frac + val_frac))

    # Fit scaler on train only
    X_train_flat = X_seq_raw[:t1].reshape(-1, X_seq_raw.shape[2])
    scaler = MinMaxScaler()
    scaler.fit(X_train_flat)

    # Scale all sequences
    X_seq = np.zeros_like(X_seq_raw)
    for i in range(n_seq):
        X_seq[i] = scaler.transform(X_seq_raw[i])

    # Scaled targets
    full_scaled = scaler.transform(X_raw)
    price_col_index = 0  # P(T) is first
    y_out_scaled = full_scaled[lookback:, price_col_index].astype(np.float32)

    return {
        "X_seq": X_seq,
        "y_out_scaled": y_out_scaled,
        "y_raw": y_raw,
        "X_raw": X_raw,
        "scaler": scaler,
        "n_features": X_seq.shape[2],
        "lookback": lookback,
        "t1": t1,
        "t2": t2,
        "n_seq": n_seq,
        "price_col_index": price_col_index,
        "valid_index": X_df.index,
        "lstm_features": lstm_features,
    }


def inverse_transform_price(scaled_price_1d, scaler, n_features, price_col_index=0):
    """Inverse transform price only (same as v2/train_lstm.py)."""
    n = len(scaled_price_1d)
    dummy = np.zeros((n, n_features), dtype=np.float32)
    dummy[:, price_col_index] = np.asarray(scaled_price_1d).ravel()
    return scaler.inverse_transform(dummy)[:, price_col_index]


def load_lstm_model(config, base_dir, n_features):
    """Load trained LSTM model."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required for LSTM inference")

    model_path = resolve_path(config["models"]["lstm"]["model_path"], base_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Load v2 lstm_config for model architecture
    lstm_config_path = V2_ROOT / "lstm_config.yaml"
    with open(lstm_config_path, "r") as f:
        lstm_config = yaml.safe_load(f)

    model = build_model(lstm_config, n_features=ckpt.get("n_features", n_features))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


def get_baseline_features():
    """Return baseline feature list (same as src/models/baseline.py)."""
    original_features = [
        'L(T-1)', 'L(T-2)', 'L(T-24)', 'L(T-48)',
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)',
        'Day', 'Season'
    ]
    calendar_features = ['Hour', 'DayOfWeek', 'Month']
    return original_features + calendar_features


def get_curated_19_features():
    """Return curated 19-feature set (from create_dataset_cleaned_19.py)."""
    return [
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)', 'P_T-168', 'Price_MA_24h',
        'Hour', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsMonday', 'IsWeekend', 'Month', 'Season',
        'CDH',
        'Net_Load', 'Solar_Ramp',
        'Load_MA_24h',
        'Hour_x_CDH',
    ]


def load_xgb_model(config, base_dir):
    """Load trained XGBoost model."""
    model_path_str = config["models"]["xgboost"].get("model_path")
    if model_path_str is None:
        raise FileNotFoundError("XGBoost model_path is null - will train on the fly")

    model_path = resolve_path(model_path_str, base_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {model_path}\n"
            "Run train_baseline.py or train_enhanced.py first, or update model_path in config."
        )

    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data.get("feature_names")

    # If feature_names is None, try to get from model or infer from n_features
    if feature_names is None:
        feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        # Infer from model's expected input size
        n_features = getattr(model, "n_features_in_", None)
        if n_features == 19:
            feature_names = get_curated_19_features()
            print(f"  Warning: No feature_names in model, using curated 19 features")
        elif n_features == 13:
            feature_names = get_baseline_features()
            print(f"  Warning: No feature_names in model, using baseline 13 features")
        else:
            # Default to baseline
            feature_names = get_baseline_features()
            print(f"  Warning: No feature_names in model (expected {n_features}), using baseline defaults")

    return model, feature_names


def predict_lstm(model, X_seq, scaler, n_features, price_col_index):
    """Generate LSTM predictions in original price space."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_seq)
        y_pred_scaled = model(X_tensor).numpy().flatten()
    y_pred = inverse_transform_price(y_pred_scaled, scaler, n_features, price_col_index)
    return y_pred


def predict_xgb(model, df, feature_names, test_timestamps):
    """Generate XGBoost predictions for test timestamps."""
    # Build feature matrix for test timestamps
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = set(feature_names) - set(available_features)

    if missing_features:
        print(f"  Warning: XGB features missing in dataset (filled with 0): {sorted(missing_features)}")

    X_test = pd.DataFrame(index=test_timestamps)
    for f in feature_names:
        if f in df.columns:
            X_test[f] = df.loc[test_timestamps, f].values
        else:
            X_test[f] = 0.0

    # Fill any NaN
    X_test = X_test.fillna(0)

    y_pred = model.predict(X_test[feature_names].values)
    return y_pred


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """Calculate MAPE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def tune_weights(y_val_actual, y_val_xgb, y_val_lstm, alpha_grid):
    """Find best alpha on validation set."""
    best_alpha = 0.5
    best_mape = float("inf")

    for alpha in alpha_grid:
        y_ensemble = alpha * y_val_xgb + (1 - alpha) * y_val_lstm
        mape = calculate_mape(y_val_actual, y_ensemble)
        if mape < best_mape:
            best_mape = mape
            best_alpha = alpha

    return best_alpha, best_mape


def main():
    parser = argparse.ArgumentParser(description="XGBoost + LSTM Ensemble Prediction")
    parser.add_argument("--config", default=None, help="Path to ensemble_config.yaml")
    parser.add_argument("--tune-weights", action="store_true", help="Tune weights on validation set")
    parser.add_argument("--out", default="results", help="Output directory for predictions")
    args = parser.parse_args()

    print("=" * 60)
    print("XGBOOST + LSTM ENSEMBLE PREDICTION")
    print("=" * 60)

    # Load config
    config, base_dir = load_config(args.config)
    out_dir = V3_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override tune_weights from CLI if specified
    do_tune = args.tune_weights or config["ensemble"].get("tune_weights", False)

    # Load data
    print("\n1. Loading data...")
    df = load_data(config, base_dir)
    print(f"   Rows: {len(df)}, Date range: {df.index.min()} to {df.index.max()}")

    # Prepare LSTM data
    print("\n2. Preparing LSTM sequences...")
    lstm_data = prepare_lstm_data(df, config)
    t1, t2, n_seq = lstm_data["t1"], lstm_data["t2"], lstm_data["n_seq"]
    lookback = lstm_data["lookback"]
    print(f"   Sequences: {n_seq}, lookback: {lookback}")
    print(f"   Split: train={t1}, val={t2-t1}, test={n_seq-t2}")

    # Derive test and validation timestamps
    valid_index = lstm_data["valid_index"]
    # Test timestamps: rows [lookback + t2, lookback + n_seq) in the valid dataframe
    test_start_row = lookback + t2
    test_end_row = lookback + n_seq
    if test_end_row > len(valid_index):
        test_end_row = len(valid_index)
    test_timestamps = valid_index[test_start_row:test_end_row]

    # Validation timestamps
    val_start_row = lookback + t1
    val_end_row = lookback + t2
    val_timestamps = valid_index[val_start_row:val_end_row]

    # Actual prices
    y_raw = lstm_data["y_raw"]
    y_test_actual = y_raw[test_start_row:test_end_row]
    y_val_actual = y_raw[val_start_row:val_end_row]

    # Load LSTM model
    print("\n3. Loading LSTM model...")
    try:
        lstm_model = load_lstm_model(config, base_dir, lstm_data["n_features"])
        print(f"   LSTM loaded from {config['models']['lstm']['model_path']}")
    except Exception as e:
        print(f"   ERROR loading LSTM: {e}")
        sys.exit(1)

    # LSTM predictions
    X_test_lstm = lstm_data["X_seq"][t2:n_seq]
    X_val_lstm = lstm_data["X_seq"][t1:t2]
    y_test_lstm = predict_lstm(
        lstm_model, X_test_lstm, lstm_data["scaler"],
        lstm_data["n_features"], lstm_data["price_col_index"]
    )
    y_val_lstm = predict_lstm(
        lstm_model, X_val_lstm, lstm_data["scaler"],
        lstm_data["n_features"], lstm_data["price_col_index"]
    )

    # Load XGBoost model
    print("\n4. Loading XGBoost model...")
    try:
        xgb_model, xgb_features = load_xgb_model(config, base_dir)
        print(f"   XGBoost loaded from {config['models']['xgboost']['model_path']}")
        print(f"   XGBoost features: {len(xgb_features)}")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print("\n   Falling back to training XGBoost on the fly (same test period)...")
        # Train XGB on the fly as fallback with curated 19 features
        xgb_features = get_curated_19_features()
        xgb_features = [f for f in xgb_features if f in df.columns]
        if len(xgb_features) < 10:
            # Fallback to basic features if curated not available
            xgb_features = [
                "P(T-1)", "P(T-2)", "P(T-24)", "P(T-48)",
                "Hour", "DayOfWeek", "Month", "Season",
            ]
            xgb_features = [f for f in xgb_features if f in df.columns]
        if not xgb_features:
            print("   ERROR: No XGB features found in dataset.")
            sys.exit(1)

        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("   ERROR: xgboost not installed.")
            sys.exit(1)

        # Train on data up to test start
        train_end_row = test_start_row
        X_train_xgb = df.iloc[:train_end_row][xgb_features].fillna(0)
        y_train_xgb = df.iloc[:train_end_row][config["data"]["target"]]
        xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train_xgb, y_train_xgb)
        print(f"   XGBoost trained on {len(X_train_xgb)} rows with {len(xgb_features)} features")

    # XGBoost predictions
    y_test_xgb = predict_xgb(xgb_model, df, xgb_features, test_timestamps)
    y_val_xgb = predict_xgb(xgb_model, df, xgb_features, val_timestamps)

    # Align lengths (should match, but just in case)
    min_test_len = min(len(y_test_actual), len(y_test_lstm), len(y_test_xgb))
    y_test_actual = y_test_actual[:min_test_len]
    y_test_lstm = y_test_lstm[:min_test_len]
    y_test_xgb = y_test_xgb[:min_test_len]
    test_timestamps = test_timestamps[:min_test_len]

    min_val_len = min(len(y_val_actual), len(y_val_lstm), len(y_val_xgb))
    y_val_actual = y_val_actual[:min_val_len]
    y_val_lstm = y_val_lstm[:min_val_len]
    y_val_xgb = y_val_xgb[:min_val_len]

    # Tune weights on validation set
    print("\n5. Combining predictions...")
    if do_tune:
        alpha_grid = config["ensemble"].get("alpha_grid", [0.5, 0.6, 0.7, 0.8, 0.9])
        best_alpha, val_ensemble_mape = tune_weights(y_val_actual, y_val_xgb, y_val_lstm, alpha_grid)
        print(f"   Weight tuning (validation): best alpha={best_alpha:.2f}, val ensemble MAPE={val_ensemble_mape:.2f}%")
    else:
        best_alpha = config["ensemble"]["weights"]["xgboost"]
        print(f"   Using config weights: alpha={best_alpha:.2f}")

    # Ensemble prediction
    y_test_ensemble = best_alpha * y_test_xgb + (1 - best_alpha) * y_test_lstm

    # Metrics
    xgb_mape = calculate_mape(y_test_actual, y_test_xgb)
    lstm_mape = calculate_mape(y_test_actual, y_test_lstm)
    ensemble_mape = calculate_mape(y_test_actual, y_test_ensemble)

    xgb_rmse = calculate_rmse(y_test_actual, y_test_xgb)
    lstm_rmse = calculate_rmse(y_test_actual, y_test_lstm)
    ensemble_rmse = calculate_rmse(y_test_actual, y_test_ensemble)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"{'Model':<15} {'MAPE %':>10} {'RMSE':>12}")
    print("-" * 40)
    print(f"{'XGBoost':<15} {xgb_mape:>10.2f} {xgb_rmse:>12.2f}")
    print(f"{'LSTM':<15} {lstm_mape:>10.2f} {lstm_rmse:>12.2f}")
    print(f"{'Ensemble':<15} {ensemble_mape:>10.2f} {ensemble_rmse:>12.2f}")
    print("-" * 40)
    print(f"Optimal weights: XGBoost={best_alpha:.2f}, LSTM={1-best_alpha:.2f}")
    print("=" * 60)

    # Save predictions
    predictions_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual": y_test_actual,
        "xgb_pred": y_test_xgb,
        "lstm_pred": y_test_lstm,
        "ensemble_pred": y_test_ensemble,
    })
    predictions_path = out_dir / "ensemble_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved: {predictions_path}")

    # Save summary
    summary = {
        "xgb_mape": float(xgb_mape),
        "lstm_mape": float(lstm_mape),
        "ensemble_mape": float(ensemble_mape),
        "xgb_rmse": float(xgb_rmse),
        "lstm_rmse": float(lstm_rmse),
        "ensemble_rmse": float(ensemble_rmse),
        "optimal_alpha": float(best_alpha),
        "test_samples": int(min_test_len),
    }
    summary_path = out_dir / "ensemble_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"Summary saved: {summary_path}")

    return summary


if __name__ == "__main__":
    main()
