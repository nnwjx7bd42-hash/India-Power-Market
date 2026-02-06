#!/usr/bin/env python3
"""
V4 — End-to-end probabilistic planning pipeline.

1. Load planning dataset + trained quantile model
2. Predict quantiles for the holdout week (168 hours)
3. Apply conformal calibration
4. Build rank-correlation matrix from historical residuals
5. Generate 200 raw scenarios via Gaussian copula
6. Reduce to 10 representative scenarios
7. Save everything + plot fan chart

Usage:
  python run_planning.py [--config config/planning_config.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

V4_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V4_ROOT))

from models.quantile_xgb import QuantileForecaster
from models.conformal_wrapper import ConformalCalibrator
from scenarios.copula_generator import (
    build_weekly_residuals,
    estimate_rank_correlation,
    generate_scenarios,
)
from scenarios.scenario_reduction import forward_reduction
from scenarios.scenario_diagnostics import plot_fan_chart, plot_spread_distribution, plot_correlation_check
from evaluation.calibration_plot import reliability_diagram, pit_histogram, print_calibration_table


def load_config(path=None):
    if path is None:
        path = V4_ROOT / "config" / "planning_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="V4 end-to-end planning pipeline")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = V4_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V4 — PROBABILISTIC PLANNING PIPELINE")
    print("=" * 60)

    # ----------------------------------------------------------------- 1. Load
    print("\n1. Loading planning dataset ...")
    dataset_path = V4_ROOT / "data" / "planning_dataset.parquet"
    df = pd.read_parquet(dataset_path)
    target = config["data"]["target"]

    feat_cfg = config["features"]
    feature_names = (
        feat_cfg["calendar"] + feat_cfg["weather"] + feat_cfg["load"]
        + feat_cfg["anchors"] + feat_cfg["interactions"]
    )
    feature_names = [f for f in feature_names if f in df.columns]
    quantiles = np.array(config["quantiles"])

    tz = df.index.tz
    train_end = pd.Timestamp(config["data"]["train_end"], tz=tz)
    val_end = pd.Timestamp(config["data"]["val_end"], tz=tz)
    holdout_hours = config["data"].get("holdout_hours", 168)
    holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours - 1)

    df_train = df[df.index <= train_end]
    df_val = df[(df.index > train_end) & (df.index <= val_end)]
    df_holdout = df[df.index >= holdout_start]

    X_holdout = df_holdout[feature_names].values
    y_holdout = df_holdout[target].values
    print(f"   Holdout: {len(df_holdout)} hours ({df_holdout.index.min()} → {df_holdout.index.max()})")

    # ----------------------------------------------------------------- 2. Load model & predict
    print("\n2. Loading quantile model and predicting ...")
    model = QuantileForecaster.load(out_dir)
    q_forecasts = model.predict(X_holdout, feature_names=feature_names)
    print(f"   Quantile forecasts shape: {q_forecasts.shape}")

    # ----------------------------------------------------------------- 3. Conformal calibration
    print("\n3. Conformal calibration ...")
    cal_weeks = config.get("conformal", {}).get("calibration_weeks", 4)
    cal_hours = cal_weeks * 168
    # Use last cal_weeks of validation as calibration set
    df_cal = df_val.iloc[-cal_hours:] if len(df_val) >= cal_hours else df_val
    X_cal = df_cal[feature_names].values
    y_cal = df_cal[target].values
    q_cal = model.predict(X_cal, feature_names=feature_names)

    calibrator = ConformalCalibrator(
        quantiles=quantiles.tolist(),
        target_coverage=config.get("conformal", {}).get("target_coverage", 0.90),
        learning_rate=config.get("conformal", {}).get("learning_rate", 0.05),
    )
    cal_report = calibrator.fit(y_cal, q_cal)
    print("   Calibration report:")
    for k, v in cal_report.items():
        print(f"     {k}: {v:.4f}")

    q_adjusted = calibrator.adjust(q_forecasts)
    print(f"   Adjusted quantile forecasts shape: {q_adjusted.shape}")

    # Save conformal state
    with open(out_dir / "conformal_state.json", "w") as f:
        json.dump(calibrator.state_dict(), f, indent=2)

    # ----------------------------------------------------------------- 4. Calibration diagnostics
    print("\n4. Calibration diagnostics ...")
    reliability_diagram(
        y_holdout, q_adjusted, quantiles,
        title="Holdout Reliability Diagram (post-conformal)",
        save_path=out_dir / "reliability_diagram.png",
    )
    pit_histogram(
        y_holdout, q_adjusted, quantiles,
        title="Holdout PIT Histogram (post-conformal)",
        save_path=out_dir / "pit_histogram.png",
    )
    print_calibration_table(y_holdout, q_adjusted, quantiles)

    # ----------------------------------------------------------------- 5. Build correlation matrix
    print("\n5. Building rank correlation matrix from historical residuals ...")
    # Get median predictions on training data for residual computation
    X_train_all = df_train[feature_names].values
    y_train_all = df_train[target].values
    q_train = model.predict(X_train_all, feature_names=feature_names)
    median_idx = int(np.argmin(np.abs(quantiles - 0.50)))
    y_median_train = q_train[:, median_idx]

    corr_window = config.get("scenarios", {}).get("correlation_window_weeks", 52)
    # Use last N weeks of training for correlation
    max_hours = corr_window * 168
    if len(y_train_all) > max_hours:
        y_act_window = y_train_all[-max_hours:]
        y_med_window = y_median_train[-max_hours:]
    else:
        y_act_window = y_train_all
        y_med_window = y_median_train

    weekly_resid = build_weekly_residuals(y_act_window, y_med_window)
    print(f"   Residual matrix: {weekly_resid.shape} (weeks x hours)")
    corr_matrix = estimate_rank_correlation(weekly_resid)
    print(f"   Correlation matrix: {corr_matrix.shape}")
    np.save(out_dir / "correlation_matrix.npy", corr_matrix)

    # ----------------------------------------------------------------- 6. Generate scenarios
    print("\n6. Generating scenarios via Gaussian copula ...")
    n_raw = config.get("scenarios", {}).get("n_raw", 200)
    seed = config.get("scenarios", {}).get("seed", 42)

    raw_scenarios = generate_scenarios(
        q_forecasts=q_adjusted,
        quantiles=quantiles,
        corr_matrix=corr_matrix,
        n_scenarios=n_raw,
        seed=seed,
    )
    print(f"   Raw scenarios: {raw_scenarios.shape}")

    # ----------------------------------------------------------------- 7. Reduce scenarios
    print("\n7. Reducing scenarios ...")
    n_reduced = config.get("scenarios", {}).get("n_reduced", 10)
    reduced_scenarios, weights = forward_reduction(raw_scenarios, n_keep=n_reduced)
    print(f"   Reduced: {reduced_scenarios.shape}, weights sum: {weights.sum():.4f}")

    np.save(out_dir / "scenarios_raw.npy", raw_scenarios)
    np.save(out_dir / "scenarios_reduced.npy", reduced_scenarios)
    np.save(out_dir / "scenario_weights.npy", weights)

    # Save reduced scenarios as CSV for easy inspection
    sc_df = pd.DataFrame(
        reduced_scenarios.T,
        index=df_holdout.index[:reduced_scenarios.shape[1]],
        columns=[f"scenario_{i+1}" for i in range(n_reduced)],
    )
    sc_df.insert(0, "actual", y_holdout[:reduced_scenarios.shape[1]])
    sc_df.to_csv(out_dir / "reduced_scenarios.csv")

    # ----------------------------------------------------------------- 8. Diagnostics plots
    print("\n8. Plotting diagnostics ...")
    plot_fan_chart(
        raw_scenarios,
        quantile_forecasts=q_adjusted,
        quantiles=quantiles,
        actuals=y_holdout,
        timestamps=df_holdout.index,
        title=f"V4 Planning — {n_raw} Scenarios (holdout week)",
        save_path=out_dir / "fan_chart.png",
    )
    plot_spread_distribution(
        raw_scenarios,
        historical_prices=y_train_all[-168*8:],  # last 8 weeks
        title="Daily Spread: Scenarios vs Historical",
        save_path=out_dir / "spread_distribution.png",
    )
    corr_mae = plot_correlation_check(
        raw_scenarios,
        corr_matrix,
        title="Scenario Correlation Quality",
        save_path=out_dir / "correlation_check.png",
    )

    # ----------------------------------------------------------------- 9. Summary
    print("\n9. Saving summary ...")
    # Scenario stats
    sc_mean = float(np.mean(raw_scenarios))
    sc_std = float(np.std(raw_scenarios))
    daily_spreads = []
    for s in range(raw_scenarios.shape[0]):
        for d in range(7):
            day = raw_scenarios[s, d*24:(d+1)*24]
            daily_spreads.append(float(day.max() - day.min()))

    summary = {
        "holdout_start": str(df_holdout.index.min()),
        "holdout_end": str(df_holdout.index.max()),
        "holdout_hours": int(len(df_holdout)),
        "n_raw_scenarios": int(n_raw),
        "n_reduced_scenarios": int(n_reduced),
        "scenario_mean_price": round(sc_mean, 2),
        "scenario_std_price": round(sc_std, 2),
        "scenario_daily_spread_mean": round(float(np.mean(daily_spreads)), 2),
        "scenario_daily_spread_p95": round(float(np.percentile(daily_spreads, 95)), 2),
        "correlation_mae": round(corr_mae, 4),
        "reduced_scenario_weights": [round(float(w), 4) for w in weights],
    }
    with open(out_dir / "planning_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs in: {out_dir}/")
    print("  quantile_model.json          — trained XGBoost booster")
    print("  conformal_state.json         — calibration corrections")
    print("  scenarios_raw.npy            — 200 copula scenarios")
    print("  scenarios_reduced.npy        — 10 representative scenarios")
    print("  scenario_weights.npy         — probability weights")
    print("  reduced_scenarios.csv        — reduced scenarios (readable)")
    print("  fan_chart.png                — scenario fan chart")
    print("  reliability_diagram.png      — calibration quality")
    print("  pit_histogram.png            — PIT uniformity check")
    print("  spread_distribution.png      — daily spread comparison")
    print("  correlation_check.png        — correlation quality")
    print("  planning_summary.yaml        — summary statistics")
    print("\nDone.")


if __name__ == "__main__":
    main()
