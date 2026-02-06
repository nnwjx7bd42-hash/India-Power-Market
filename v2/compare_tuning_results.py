#!/usr/bin/env python3
"""
Compare single-split vs nested CV LSTM tuning results.
Reads trials CSVs (best val/CV MAPE) and optional metrics YAMLs (test MAPE).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def get_v2_root():
    return Path(__file__).resolve().parent


def resolve_path(path, root):
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p


def load_best_value_from_trials(csv_path):
    """Load trials CSV, filter COMPLETE, return min value (best MAPE %) or None."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if "state" in df.columns:
            df = df[df["state"] == "COMPLETE"]
        if "value" not in df.columns or len(df) == 0:
            return None
        return float(df["value"].min())
    except Exception:
        return None


def load_metrics_yaml(yaml_path):
    """Load YAML with test_mape (and optionally test_rmse). Return dict or None."""
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare single-split vs nested CV LSTM tuning results"
    )
    parser.add_argument(
        "--single-trials",
        default="results/optuna_lstm_trials.csv",
        help="Single-split trials CSV (default: results/optuna_lstm_trials.csv)",
    )
    parser.add_argument(
        "--nested-trials",
        default="results/optuna_lstm_nested_cv_trials.csv",
        help="Nested CV trials CSV (default: results/optuna_lstm_nested_cv_trials.csv)",
    )
    parser.add_argument(
        "--metrics",
        default="results/lstm_metrics.yaml",
        help="Current model metrics YAML (default: results/lstm_metrics.yaml)",
    )
    parser.add_argument(
        "--single-metrics",
        default=None,
        help="YAML with test_mape for single-split-trained model (optional)",
    )
    parser.add_argument(
        "--nested-metrics",
        default=None,
        help="YAML with test_mape for nested-CV-trained model (optional)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write summary to this file (e.g. results/tuning_comparison.txt or .json)",
    )
    args = parser.parse_args()

    root = get_v2_root()
    single_trials_path = resolve_path(args.single_trials, root)
    nested_trials_path = resolve_path(args.nested_trials, root)
    metrics_path = resolve_path(args.metrics, root)
    single_metrics_path = resolve_path(args.single_metrics, root) if args.single_metrics else None
    nested_metrics_path = resolve_path(args.nested_metrics, root) if args.nested_metrics else None

    single_val_mape = load_best_value_from_trials(single_trials_path)
    nested_cv_mape = load_best_value_from_trials(nested_trials_path)
    current_metrics = load_metrics_yaml(metrics_path)
    current_test_mape = float(current_metrics["test_mape"]) if current_metrics and "test_mape" in current_metrics else None
    current_test_rmse = current_metrics.get("test_rmse") if current_metrics else None

    single_metrics = load_metrics_yaml(single_metrics_path) if single_metrics_path else None
    nested_metrics = load_metrics_yaml(nested_metrics_path) if nested_metrics_path else None
    single_test_mape = float(single_metrics["test_mape"]) if single_metrics and "test_mape" in single_metrics else None
    nested_test_mape = float(nested_metrics["test_mape"]) if nested_metrics and "test_mape" in nested_metrics else None

    lines = []
    lines.append("=" * 60)
    lines.append("LSTM TUNING RESULTS COMPARISON")
    lines.append("=" * 60)

    if single_val_mape is not None:
        lines.append(f"Single-split tuning   | Best val MAPE: {single_val_mape:.2f}%")
    else:
        lines.append("Single-split tuning   | (no trials CSV or no complete trials)")

    if nested_cv_mape is not None:
        lines.append(f"Nested CV tuning      | Best mean CV MAPE: {nested_cv_mape:.2f}%")
    else:
        lines.append("Nested CV tuning      | (no trials CSV or no complete trials)")

    if current_test_mape is not None:
        s = f"Current model          | Test MAPE: {current_test_mape:.2f}%"
        if current_test_rmse is not None:
            s += f"  RMSE: {current_test_rmse:.2f}"
        s += "  (from lstm_metrics.yaml)"
        lines.append(s)
    else:
        lines.append("Current model          | (no lstm_metrics.yaml or no test_mape)")

    if single_metrics_path and nested_metrics_path and single_test_mape is not None and nested_test_mape is not None:
        lines.append("")
        lines.append("--- With --single-metrics and --nested-metrics ---")
        lines.append(f"Single-split model    | Test MAPE: {single_test_mape:.2f}%")
        if single_val_mape is not None:
            gap = single_val_mape - single_test_mape
            lines.append(f"                       | Val-Test gap: {gap:+.2f} pp")
        lines.append(f"Nested CV model       | Test MAPE: {nested_test_mape:.2f}%")
        if nested_cv_mape is not None:
            gap = nested_cv_mape - nested_test_mape
            lines.append(f"                       | CV-Test gap: {gap:+.2f} pp")

    lines.append("")
    for line in lines:
        print(line)

    summary = {
        "single_split_best_val_mape": single_val_mape,
        "nested_cv_best_mean_cv_mape": nested_cv_mape,
        "current_model_test_mape": current_test_mape,
        "current_model_test_rmse": current_test_rmse,
    }
    if single_test_mape is not None:
        summary["single_split_model_test_mape"] = single_test_mape
        if single_val_mape is not None:
            summary["single_split_val_test_gap_pp"] = round(single_val_mape - single_test_mape, 2)
    if nested_test_mape is not None:
        summary["nested_cv_model_test_mape"] = nested_test_mape
        if nested_cv_mape is not None:
            summary["nested_cv_cv_test_gap_pp"] = round(nested_cv_mape - nested_test_mape, 2)

    if args.out:
        out_path = resolve_path(args.out, root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
        else:
            with open(out_path, "w") as f:
                f.write("\n".join(lines) + "\n")
        print(f"Summary written to: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
