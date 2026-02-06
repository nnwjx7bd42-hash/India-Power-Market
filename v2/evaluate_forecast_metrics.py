#!/usr/bin/env python3
"""
Evaluate holdout forecasts with multi-dimensional metrics:

- Hit rate (exact match %)
- Accuracy within tolerance bands (±5%, ±10%, beyond ±10%)
- Mean Directional Accuracy (MDA) – direction of change vs previous actual
- MAPE, MAE (magnitude)

Reads holdout_actuals_and_predictions.xlsx or holdout_predictions.csv and writes
holdout_forecast_metrics.yaml.

Usage:
  python evaluate_forecast_metrics.py [--predictions results/holdout_predictions.csv] [--out results]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Use project src for metrics
V2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V2_ROOT.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from validation.metrics import forecast_metrics_report


def load_predictions(path: Path) -> pd.DataFrame:
    """Load actuals and predictions from CSV or Excel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .csv or .xlsx")

    if "actual" not in df.columns:
        raise ValueError("Data must contain an 'actual' column")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate holdout forecasts: hit rate, tolerance bands, MDA, MAPE, MAE"
    )
    parser.add_argument(
        "--predictions",
        default=V2_ROOT / "results" / "holdout_predictions.csv",
        help="Path to holdout CSV or xlsx (timestamp, actual, lstm_pred, xgb_pred, ...)",
    )
    parser.add_argument(
        "--out",
        default=V2_ROOT / "results",
        help="Output directory for holdout_forecast_metrics.yaml",
    )
    parser.add_argument(
        "--tolerance",
        default="5,10",
        help="Tolerance bands in percent, comma-separated (default: 5,10)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = V2_ROOT / predictions_path

    tolerance_pct = tuple(float(x.strip()) for x in args.tolerance.split(","))

    print("Loading predictions...")
    df = load_predictions(predictions_path)

    y_true = df["actual"].values
    pred_columns = [c for c in df.columns if c not in ("timestamp", "actual")]

    if not pred_columns:
        print("ERROR: No prediction columns found (expected e.g. lstm_pred, xgb_pred).")
        sys.exit(1)

    n_periods = len(y_true)
    print(f"Periods: {n_periods}")
    print(f"Models:  {pred_columns}")

    results = {
        "n_periods": int(n_periods),
        "tolerance_pct": list(tolerance_pct),
        "metrics": {},
    }

    for col in pred_columns:
        y_pred = df[col].values
        report = forecast_metrics_report(
            y_true, y_pred, tolerance_pct=tolerance_pct, include_magnitude=True
        )
        results["metrics"][col] = report

        print(f"\n--- {col} ---")
        print(f"  Hit rate:        {report['hit_rate_pct']:.2f}%")
        for k in sorted(report.keys()):
            if (k.startswith("within_") or k.startswith("beyond_")) and "_n_" not in k and isinstance(report[k], (int, float)):
                print(f"  {k}: {report[k]:.2f}%")
        print(f"  Entire period:   forecast > actual: {report['n_over_actual']} times, forecast < actual: {report['n_under_actual']} times")
        n5 = report.get("within_5pct_n_total") or 0
        if n5 > 0:
            print(f"  Within ±5%:      {n5} periods → forecast > actual: {report.get('within_5pct_n_over', 0)}, forecast < actual: {report.get('within_5pct_n_under', 0)}, exact: {report.get('within_5pct_n_exact', 0)}")
        print(f"  MDA:             {report['mda_pct']:.2f}%")
        print(f"  MAPE:            {report['mape']:.2f}%")
        print(f"  MAE:             {report['mae']:.2f}")

    out_path = out_dir / "holdout_forecast_metrics.yaml"
    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print(f"\nMetrics saved: {out_path}")
    return results


if __name__ == "__main__":
    main()
