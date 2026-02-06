#!/usr/bin/env python3
"""
Bayesian hyperparameter tuning for v2 LSTM using Optuna (TPE sampler).
Reuses data loading, sequence building, scaling, and model from train_lstm.py.
Minimizes validation MAPE (inverse-transformed to original price space).
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import MinMaxScaler
import yaml

# Reuse train_lstm pipeline and backend
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

if HAS_TF:
    from tensorflow import keras
    from tensorflow.keras import layers
if HAS_TORCH:
    import torch
    import torch.nn as nn


def _train_pytorch(model, X_train, y_train, X_val, y_val, trial_config, device):
    """Train LSTM (PyTorch) with early stopping; return trained model (best weights)."""
    lr = trial_config["training"]["learning_rate"]
    batch_size = trial_config["training"]["batch_size"]
    epochs = trial_config["training"].get("epochs", 50)
    patience = trial_config["training"].get("early_stopping_patience", 10)
    reduce_lr_patience = trial_config["training"].get("reduce_lr_patience")
    reduce_lr_factor = trial_config["training"].get("reduce_lr_factor", 0.5)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if reduce_lr_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=reduce_lr_factor, patience=reduce_lr_patience
        )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_train_t))
        for start in range(0, len(X_train_t), batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.functional.mse_loss(val_pred, y_val_t).item()
        if scheduler:
            scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _train_tensorflow(model, X_train, y_train, X_val, y_val, trial_config):
    """Train LSTM (TensorFlow) with early stopping; return trained model (best weights)."""
    lr = trial_config["training"]["learning_rate"]
    batch_size = trial_config["training"]["batch_size"]
    epochs = trial_config["training"].get("epochs", 50)
    patience = trial_config["training"].get("early_stopping_patience", 10)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    reduce_lr = trial_config["training"].get("reduce_lr_patience")
    if reduce_lr is not None:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=trial_config["training"].get("reduce_lr_factor", 0.5),
                patience=reduce_lr,
            )
        )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="LSTM Bayesian hyperparameter tuning (v2)")
    parser.add_argument("--config", default=None, help="Path to lstm_config.yaml")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    # Use local ref so nested objective does not shadow module (UnboundLocalError)
    opt = optuna

    if not HAS_TF and not HAS_TORCH:
        print("Install TensorFlow or PyTorch: pip install tensorflow | pip install torch")
        sys.exit(1)
    backend = "tensorflow" if HAS_TF else "pytorch"
    print(f"Backend: {backend}")

    root = get_v2_root()
    config = load_config(args.config)
    train_frac = config["data"]["train_frac"]
    val_frac = config["data"]["val_frac"]
    test_frac = config["data"]["test_frac"]
    price_col_index = 0
    target_col = config["data"]["target"]

    # Load data once (same as train_lstm)
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

    def objective(trial) -> float:
        lookback = trial.suggest_categorical("lookback", [24, 48, 72, 168])
        lstm_units_1 = trial.suggest_categorical("lstm_units_1", [32, 50, 64, 100])
        lstm_units_2 = trial.suggest_categorical("lstm_units_2", [16, 30, 50])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])

        trial_config = copy.deepcopy(config)
        trial_config["sequence"]["lookback_hours"] = lookback
        trial_config["model"]["lstm_units"] = [lstm_units_1, lstm_units_2]
        trial_config["model"]["dropout"] = dropout
        trial_config["model"]["bidirectional"] = bidirectional
        trial_config["training"]["learning_rate"] = learning_rate
        trial_config["training"]["batch_size"] = batch_size

        X_seq_raw, y_out = build_sequences(X_raw, y_raw, lookback)
        n_seq = len(y_out)
        if n_seq < 100:
            return 1e6  # penalty if too few sequences
        t1 = int(n_seq * train_frac)
        t2 = int(n_seq * (train_frac + val_frac))
        train_idx = slice(0, t1)
        val_idx = slice(t1, t2)

        X_train_flat = X_seq_raw[train_idx].reshape(-1, X_seq_raw.shape[2])
        scaler = MinMaxScaler()
        scaler.fit(X_train_flat)
        X_seq = np.zeros_like(X_seq_raw)
        for i in range(n_seq):
            X_seq[i] = scaler.transform(X_seq_raw[i])
        full_scaled = scaler.transform(X_raw)
        y_out_scaled = full_scaled[lookback:, price_col_index].astype(np.float32)

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
        mape = calculate_mape(y_val_actual, y_pred_actual)
        return float(mape)

    print("\nStarting Bayesian optimization (TPE)...")
    print("=" * 60)
    study = opt.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="lstm_tuning",
    )
    study.enqueue_trial(
        {
            "lookback": 24,
            "lstm_units_1": 50,
            "lstm_units_2": 30,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "bidirectional": False,
        }
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nBest validation MAPE: {study.best_value:.2f}%")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_path = out_dir / "optuna_lstm_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"\nSaved: {trials_path}")

    best_params = study.best_params
    best_config = {
        "sequence": {"lookback_hours": best_params["lookback"]},
        "model": {
            "lstm_units": [best_params["lstm_units_1"], best_params["lstm_units_2"]],
            "dropout": best_params["dropout"],
            "bidirectional": best_params["bidirectional"],
        },
        "training": {
            "learning_rate": best_params["learning_rate"],
            "batch_size": best_params["batch_size"],
        },
    }
    best_path = out_dir / "optuna_lstm_best_params.yaml"
    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {best_path}")

    try:
        import optuna.importance as optuna_importance
        importances = optuna_importance.get_param_importances(study)
        print("\nHyperparameter importance:")
        for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {param}: {imp:.3f}")
    except Exception:
        pass

    try:
        from optuna.visualization.matplotlib import plot_optimization_history  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_optimization_history(study)
        plt.savefig(out_dir / "optuna_lstm_history.png", dpi=150)
        plt.close()
        print(f"Saved: {out_dir / 'optuna_lstm_history.png'}")
    except Exception:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
