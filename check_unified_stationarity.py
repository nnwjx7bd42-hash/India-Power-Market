#!/usr/bin/env python3
"""
Check unified_dataset.parquet for stationarity and related diagnostics.
Uses Augmented Dickey-Fuller (ADF) and KPSS tests on key series.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def run_adf(series, name):
    """Augmented Dickey-Fuller: H0 = unit root (non-stationary). Reject H0 => stationary."""
    from statsmodels.tsa.stattools import adfuller
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return None, "Too few observations"
    result = adfuller(series_clean, autolag="AIC")
    adf_stat, p_value, usedlag, nobs, critical_values, icbest = result
    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "stationary_adf": p_value < 0.05,
        "critical_5pct": critical_values["5%"],
    }


def run_kpss(series, name):
    """KPSS: H0 = series is stationary. Reject H0 => non-stationary."""
    from statsmodels.tsa.stattools import kpss
    series_clean = series.dropna()
    if len(series_clean) < 20:
        return None
    try:
        stat, p_value, lags, crit = kpss(series_clean, regression="c", nlags="auto")
        return {
            "kpss_stat": stat,
            "p_value": p_value,
            "stationary_kpss": p_value > 0.05,  # Fail to reject H0 => stationary
            "critical_5pct": crit["5%"],
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    path = Path("data/processed/unified_dataset.parquet")
    if not path.exists():
        print(f"File not found: {path}")
        return

    print("=" * 70)
    print("UNIFIED DATASET: STATIONARITY & DIAGNOSTICS")
    print("=" * 70)
    df = pd.read_parquet(path)
    df = df.sort_index()

    print(f"\nFile: {path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")

    # Basic stats
    print("\n" + "-" * 70)
    print("BASIC STATISTICS (sample)")
    print("-" * 70)
    key_cols = ["P(T)", "L(T-1)", "Demand", "Thermal", "Wind", "Solar"]
    for c in key_cols:
        if c in df.columns:
            s = df[c]
            print(f"  {c}: mean={s.mean():,.2f}, std={s.std():,.2f}, min={s.min():,.2f}, max={s.max():,.2f}")

    # Stationarity tests on main series
    series_to_test = ["P(T)", "L(T-1)", "Demand", "Thermal"]
    series_to_test = [c for c in series_to_test if c in df.columns]

    print("\n" + "-" * 70)
    print("STATIONARITY TESTS")
    print("-" * 70)
    print("  ADF: H0 = unit root (non-stationary). p < 0.05 => reject H0 => STATIONARY")
    print("  KPSS: H0 = series is stationary. p > 0.05 => fail to reject => STATIONARY")
    print()

    results = []
    for col in series_to_test:
        s = df[col]
        adf = run_adf(s, col)
        kpss_res = run_kpss(s, col)
        if isinstance(adf, dict):
            adf_stationary = adf["stationary_adf"]
            adf_p = adf["p_value"]
        else:
            adf_stationary = None
            adf_p = None
        if isinstance(kpss_res, dict) and "error" not in kpss_res:
            kpss_stationary = kpss_res["stationary_kpss"]
            kpss_p = kpss_res["p_value"]
        else:
            kpss_stationary = None
            kpss_p = None

        # Overall: both tests agree on stationarity
        if adf_stationary is not None and kpss_stationary is not None:
            agree = "STATIONARY" if (adf_stationary and kpss_stationary) else "NON-STATIONARY"
        elif adf_stationary is not None:
            agree = "ADF says STATIONARY" if adf_stationary else "ADF says NON-STATIONARY"
        else:
            agree = "N/A"

        results.append((col, adf_p, kpss_p, adf_stationary, kpss_stationary, agree))
        print(f"  {col}:")
        if isinstance(adf, dict):
            print(f"    ADF  stat={adf['adf_stat']:.4f}, p-value={adf['p_value']:.6f} => {'Stationary' if adf_stationary else 'Non-stationary'}")
        if isinstance(kpss_res, dict) and "error" not in kpss_res:
            print(f"    KPSS stat={kpss_res['kpss_stat']:.4f}, p-value={kpss_res['p_value']:.6f} => {'Stationary' if kpss_stationary else 'Non-stationary'}")
        print(f"    Conclusion: {agree}")
        print()

    # Autocorrelation (first 24 lags for P(T))
    if "P(T)" in df.columns:
        print("-" * 70)
        print("AUTOCORRELATION: P(T) (first 24 lags)")
        print("-" * 70)
        s = df["P(T)"].dropna()
        acf = [s.autocorr(lag=k) for k in range(1, 25)]
        for k, r in enumerate(acf, 1):
            bar = "|" * int(abs(r) * 40) + " " * (40 - int(abs(r) * 40))
            print(f"  Lag {k:2d}: {r:+.4f} {bar}")
        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
