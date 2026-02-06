#!/usr/bin/env python3
"""
XGBoost + LSTM Ensemble Runner

CLI wrapper that runs prediction and analysis, then prints a summary.

Usage:
  python run_ensemble.py [--tune-weights] [--out results/]
  python run_ensemble.py --help
"""

import argparse
import sys
from pathlib import Path

import yaml

V3_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Run XGBoost + LSTM ensemble prediction and analysis"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to ensemble_config.yaml (default: v3/ensemble_config.yaml)",
    )
    parser.add_argument(
        "--tune-weights",
        action="store_true",
        help="Tune ensemble weights on validation set",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Output directory for predictions and plots (default: results/)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis (only run prediction)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("XGBOOST + LSTM ENSEMBLE PIPELINE")
    print("=" * 70)

    # Build command args for subprocess calls
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = V3_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Run ensemble_predict.py ---
    print("\n>>> STEP 1: PREDICTION <<<\n")

    from ensemble_predict import main as predict_main

    # Temporarily modify sys.argv for the predict script
    predict_argv = ["ensemble_predict.py", "--out", str(out_dir)]
    if args.config:
        predict_argv.extend(["--config", args.config])
    if args.tune_weights:
        predict_argv.append("--tune-weights")

    old_argv = sys.argv
    sys.argv = predict_argv
    try:
        summary = predict_main()
    finally:
        sys.argv = old_argv

    if summary is None:
        print("\nPrediction failed. Exiting.")
        sys.exit(1)

    # --- Step 2: Run ensemble_analysis.py ---
    if not args.skip_analysis:
        print("\n>>> STEP 2: ANALYSIS <<<\n")

        from ensemble_analysis import main as analysis_main

        predictions_path = out_dir / "ensemble_predictions.csv"
        analysis_argv = [
            "ensemble_analysis.py",
            "--predictions", str(predictions_path),
            "--out", str(out_dir),
        ]

        sys.argv = analysis_argv
        try:
            analysis_result = analysis_main()
        finally:
            sys.argv = old_argv

        if analysis_result:
            corr = analysis_result["correlation"]
            wins = analysis_result["ensemble_wins"]
        else:
            corr = None
            wins = None
    else:
        corr = None
        wins = None

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<15} {'MAPE %':>10} {'RMSE':>12}")
    print("-" * 40)
    print(f"{'XGBoost':<15} {summary['xgb_mape']:>10.2f} {summary['xgb_rmse']:>12.2f}")
    print(f"{'LSTM':<15} {summary['lstm_mape']:>10.2f} {summary['lstm_rmse']:>12.2f}")
    print(f"{'Ensemble':<15} {summary['ensemble_mape']:>10.2f} {summary['ensemble_rmse']:>12.2f}")
    print("-" * 40)

    alpha = summary["optimal_alpha"]
    print(f"\nOptimal weights: XGBoost = {alpha:.2f}, LSTM = {1-alpha:.2f}")

    if corr:
        print(f"Error correlation (Pearson): r = {corr['pearson_r']:.4f}")

    if wins:
        print(f"Ensemble beats both models: {wins['beats_both_pct']:.1f}% of hours")

    # Improvement check
    best_individual = min(summary["xgb_mape"], summary["lstm_mape"])
    improvement = best_individual - summary["ensemble_mape"]
    if improvement > 0:
        print(f"\nEnsemble improves over best individual by {improvement:.2f} pp")
    else:
        print(f"\nEnsemble does not improve over best individual ({-improvement:.2f} pp worse)")

    print(f"\nOutputs saved to: {out_dir}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
