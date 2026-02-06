#!/usr/bin/env python3
"""
Inspect v2 LSTM test period: date range and price volatility vs Aug 2023.
Uses same data/split as train_lstm.py (holdout 168h, train 72% / val 18% / test 10%).
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

def get_v2_root():
    return Path(__file__).resolve().parent

def load_config():
    with open(get_v2_root() / "lstm_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    root = get_v2_root()
    config = load_config()
    path = root / config["data"]["dataset"]
    if not path.exists():
        print(f"Dataset not found: {path}")
        sys.exit(1)

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        df = df.sort_index()

    holdout_hours = config["data"].get("holdout_hours", 0) or 0
    if holdout_hours > 0:
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours)
        df = df[df.index <= holdout_start]

    target = config["data"]["target"]
    if target not in df.columns:
        print(f"Target {target} not in columns")
        sys.exit(1)
    price = df[target].dropna()

    lookback = config["sequence"]["lookback_hours"]
    train_frac = config["data"]["train_frac"]
    val_frac = config["data"]["val_frac"]
    test_frac = config["data"]["test_frac"]
    n = len(price)
    n_seq = n - lookback
    t1 = int(n_seq * train_frac)
    t2 = int(n_seq * (train_frac + val_frac))
    # Test targets: sequence indices t2 : n_seq -> row indices (t2+lookback) : (n_seq-1+lookback)
    test_start_row = t2 + lookback
    test_end_row = n_seq + lookback - 1
    if test_end_row >= n:
        test_end_row = n - 1
    test_dates = price.index[test_start_row : test_end_row + 1]
    test_price = price.loc[test_dates]

    print("=" * 60)
    print("V2 LSTM TEST PERIOD INSPECTION")
    print("=" * 60)
    print(f"\nData (after {holdout_hours}h holdout): {price.index.min()} to {price.index.max()}")
    print(f"Total hours: {len(price):,}")
    print(f"Split: train 72% / val 18% / test 10%")
    print(f"\nTest period (last 10% of sequence targets):")
    print(f"  Start: {test_dates.min()}")
    print(f"  End:   {test_dates.max()}")
    print(f"  Hours: {len(test_price):,}")

    aug2023 = price.loc["2023-08-01":"2023-08-31"]
    full = price
    test = test_price

    def stats(name, s):
        if len(s) == 0:
            print(f"  {name}: (no data)")
            return
        print(f"  {name}: mean={s.mean():.0f}  std={s.std():.0f}  min={s.min():.0f}  max={s.max():.0f}  median={s.median():.0f}")

    print("\n--- Price (P(T)) volatility ---")
    print("Full series (after holdout):")
    stats("P(T)", full)
    print("\nTest period:")
    stats("P(T)", test)
    print("\nAug 2023:")
    stats("P(T)", aug2023)

    print("\n--- Comparison ---")
    if len(aug2023) > 0 and len(test) > 0:
        test_std = test.std()
        aug_std = aug2023.std()
        print(f"Test period std:  {test_std:.0f}  (volatility)")
        print(f"Aug 2023 std:     {aug_std:.0f}")
        if test_std > aug_std * 1.2:
            print("Test period is MORE volatile than Aug 2023.")
        elif test_std < aug_std * 0.8:
            print("Test period is LESS volatile than Aug 2023.")
        else:
            print("Test period volatility similar to Aug 2023.")
    print("\nNote: Aug 2023 is NOT the v2 test period; test period is the last 10% of data (see dates above).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(price.index, price.values, color="gray", alpha=0.6, linewidth=0.5, label="P(T) full")
        ax.axvspan(test_dates.min(), test_dates.max(), alpha=0.25, color="blue", label="Test period")
        ax.axvspan(pd.Timestamp("2023-08-01", tz=price.index.tz), pd.Timestamp("2023-08-31", tz=price.index.tz), alpha=0.2, color="orange", label="Aug 2023")
        ax.set_xlabel("Date")
        ax.set_ylabel("P(T) (₹/MWh)")
        ax.set_title("V2 LSTM: Test period vs Aug 2023 (price volatility)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        out = root / "results" / "test_period_inspection.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\nPlot saved: {out}")
    except Exception as e:
        print(f"\n(Plot skipped: {e})")

if __name__ == "__main__":
    main()
