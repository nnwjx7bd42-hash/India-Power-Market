#!/usr/bin/env python3
"""
XGBoost + LSTM Ensemble Analysis

Analyze ensemble predictions: error correlation, actual vs predicted plots,
residuals over time, and hours where ensemble beats both models.

Usage:
  python ensemble_analysis.py [--predictions results/ensemble_predictions.csv] [--out results/]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

V3_ROOT = Path(__file__).resolve().parent


def load_predictions(predictions_path):
    """Load ensemble predictions CSV."""
    df = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    return df


def compute_error_correlation(df):
    """Compute correlation between XGB and LSTM residuals."""
    xgb_residuals = df["actual"] - df["xgb_pred"]
    lstm_residuals = df["actual"] - df["lstm_pred"]

    pearson_r, pearson_p = stats.pearsonr(xgb_residuals, lstm_residuals)
    spearman_r, spearman_p = stats.spearmanr(xgb_residuals, lstm_residuals)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def identify_ensemble_wins(df):
    """Identify hours where ensemble beats both individual models."""
    xgb_error = np.abs(df["actual"] - df["xgb_pred"])
    lstm_error = np.abs(df["actual"] - df["lstm_pred"])
    ensemble_error = np.abs(df["actual"] - df["ensemble_pred"])

    # Ensemble beats both
    ensemble_beats_both = (ensemble_error < xgb_error) & (ensemble_error < lstm_error)

    # Ensemble beats at least one
    ensemble_beats_one = (ensemble_error < xgb_error) | (ensemble_error < lstm_error)

    return {
        "beats_both_count": ensemble_beats_both.sum(),
        "beats_both_pct": 100 * ensemble_beats_both.mean(),
        "beats_one_count": ensemble_beats_one.sum(),
        "beats_one_pct": 100 * ensemble_beats_one.mean(),
        "total_hours": len(df),
    }


def plot_actual_vs_predicted(df, out_dir):
    """Plot actual vs predicted for all models."""
    if not HAS_PLT:
        print("  matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = [
        ("XGBoost", "xgb_pred", "tab:blue"),
        ("LSTM", "lstm_pred", "tab:orange"),
        ("Ensemble", "ensemble_pred", "tab:green"),
    ]

    for ax, (name, col, color) in zip(axes, models):
        ax.scatter(df["actual"], df[col], alpha=0.3, s=10, color=color)
        # Perfect prediction line
        lims = [
            min(df["actual"].min(), df[col].min()),
            max(df["actual"].max(), df[col].max()),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Actual Price (INR/MWh)")
        ax.set_ylabel("Predicted Price (INR/MWh)")
        ax.set_title(f"{name}: Actual vs Predicted")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'actual_vs_predicted.png'}")


def plot_residuals_over_time(df, out_dir):
    """Plot residuals over time for all models."""
    if not HAS_PLT:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    models = [
        ("XGBoost", "xgb_pred", "tab:blue"),
        ("LSTM", "lstm_pred", "tab:orange"),
        ("Ensemble", "ensemble_pred", "tab:green"),
    ]

    for ax, (name, col, color) in zip(axes, models):
        residuals = df["actual"] - df[col]
        ax.plot(df["timestamp"], residuals, alpha=0.7, linewidth=0.5, color=color)
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_ylabel("Residual (INR/MWh)")
        ax.set_title(f"{name} Residuals")
        ax.fill_between(df["timestamp"], residuals, 0, alpha=0.2, color=color)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'residuals_over_time.png'}")


def plot_predictions_timeseries(df, out_dir, n_days=14):
    """Plot time series of actual and predicted prices (subset)."""
    if not HAS_PLT:
        return

    # Take last n_days for clarity
    n_hours = n_days * 24
    df_subset = df.tail(n_hours).copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_subset["timestamp"], df_subset["actual"], label="Actual", color="black", linewidth=1.5)
    ax.plot(df_subset["timestamp"], df_subset["xgb_pred"], label="XGBoost", color="tab:blue", linewidth=1, alpha=0.7)
    ax.plot(df_subset["timestamp"], df_subset["lstm_pred"], label="LSTM", color="tab:orange", linewidth=1, alpha=0.7)
    ax.plot(df_subset["timestamp"], df_subset["ensemble_pred"], label="Ensemble", color="tab:green", linewidth=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Price (INR/MWh)")
    ax.set_title(f"Price Predictions - Last {n_days} Days")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "predictions_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'predictions_timeseries.png'}")


def plot_error_comparison(df, out_dir):
    """Plot absolute error comparison between models."""
    if not HAS_PLT:
        return

    xgb_error = np.abs(df["actual"] - df["xgb_pred"])
    lstm_error = np.abs(df["actual"] - df["lstm_pred"])
    ensemble_error = np.abs(df["actual"] - df["ensemble_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    data = [xgb_error, lstm_error, ensemble_error]
    labels = ["XGBoost", "LSTM", "Ensemble"]
    bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("Absolute Error (INR/MWh)")
    axes[0].set_title("Absolute Error Distribution")

    # Histogram of improvement
    improvement = np.minimum(xgb_error, lstm_error) - ensemble_error
    axes[1].hist(improvement, bins=50, color="tab:green", alpha=0.7, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", label="No improvement")
    axes[1].axvline(improvement.mean(), color="blue", linestyle="-", label=f"Mean: {improvement.mean():.1f}")
    axes[1].set_xlabel("Error Reduction (INR/MWh)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Ensemble Improvement over Best Individual")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'error_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ensemble predictions")
    parser.add_argument(
        "--predictions",
        default="results/ensemble_predictions.csv",
        help="Path to ensemble predictions CSV",
    )
    parser.add_argument("--out", default="results", help="Output directory for plots")
    args = parser.parse_args()

    print("=" * 60)
    print("ENSEMBLE ANALYSIS")
    print("=" * 60)

    # Resolve paths
    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = V3_ROOT / predictions_path

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = V3_ROOT / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        print(f"ERROR: Predictions file not found: {predictions_path}")
        print("Run ensemble_predict.py first.")
        return None

    # Load predictions
    print("\n1. Loading predictions...")
    df = load_predictions(predictions_path)
    print(f"   Loaded {len(df)} samples")

    # Error correlation
    print("\n2. Computing error correlation...")
    corr = compute_error_correlation(df)
    print(f"   Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.4e})")
    print(f"   Spearman r: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.4e})")

    if corr["pearson_r"] < 0.5:
        print("   => Low correlation: ensemble should benefit significantly")
    elif corr["pearson_r"] < 0.7:
        print("   => Moderate correlation: ensemble may provide some benefit")
    else:
        print("   => High correlation: limited ensemble benefit expected")

    # Ensemble wins
    print("\n3. Identifying ensemble wins...")
    wins = identify_ensemble_wins(df)
    print(f"   Ensemble beats BOTH models: {wins['beats_both_count']}/{wins['total_hours']} ({wins['beats_both_pct']:.1f}%)")
    print(f"   Ensemble beats at least ONE: {wins['beats_one_count']}/{wins['total_hours']} ({wins['beats_one_pct']:.1f}%)")

    # Generate plots
    print("\n4. Generating plots...")
    plot_actual_vs_predicted(df, out_dir)
    plot_residuals_over_time(df, out_dir)
    plot_predictions_timeseries(df, out_dir)
    plot_error_comparison(df, out_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return {
        "correlation": corr,
        "ensemble_wins": wins,
    }


if __name__ == "__main__":
    main()
