#!/usr/bin/env python3
"""
Plot holdout week: actual vs LSTM vs XGBoost predictions.
Saves chart to v2/results/holdout_actual_vs_predictions.png
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "holdout_predictions.csv"
OUT_PATH = ROOT / "results" / "holdout_actual_vs_predictions.png"


def main():
    if not CSV_PATH.exists():
        print(f"Not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["timestamp"], df["actual"], label="Actual", color="black", linewidth=2, zorder=3)
    ax.plot(df["timestamp"], df["lstm_pred"], label="LSTM", color="#2563eb", linewidth=1.2, alpha=0.9)
    ax.plot(df["timestamp"], df["xgb_pred"], label="XGBoost", color="#dc2626", linewidth=1.2, alpha=0.9)

    ax.set_xlabel("Time (Jun 17–24, 2025)")
    ax.set_ylabel("Price (Rs/MWh)")
    ax.set_title("Holdout Week: Actual vs LSTM vs XGBoost Predictions")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.xticks(rotation=25)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
