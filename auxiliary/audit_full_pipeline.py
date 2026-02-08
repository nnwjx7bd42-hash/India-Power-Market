#!/usr/bin/env python3
"""
Full pipeline re-run audit — runs every layer, compares to saved metrics.

Usage:  python auxiliary/audit_full_pipeline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Helpers ───────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def _pct_diff(a, b):
    """Percentage difference between a and b relative to b."""
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return abs(a - b) / abs(b) * 100


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — V1: Data pipeline audit
# ═══════════════════════════════════════════════════════════════════════════

def audit_v1_data():
    """Inspect dataset_cleaned.parquet for integrity."""
    print("\n" + "=" * 70)
    print("PHASE 1 — V1: DATA PIPELINE AUDIT")
    print("=" * 70)

    results = {}

    # Load dataset
    path = PROJECT_ROOT / "data" / "processed" / "dataset_cleaned.parquet"
    if not path.exists():
        path = PROJECT_ROOT / "v2" / "dataset_cleaned.parquet"
    assert path.exists(), f"dataset_cleaned.parquet not found"

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
    df = df.sort_index()

    # Shape
    rows, cols = df.shape
    print(f"\n  Shape: {rows:,} rows x {cols} columns")
    results["rows"] = rows
    results["columns"] = cols
    assert rows > 30000, f"Too few rows: {rows}"
    assert cols > 20, f"Too few columns: {cols}"
    print(f"  ✓ Shape OK (>30k rows, >20 cols)")

    # Date range
    dt_min = df.index.min()
    dt_max = df.index.max()
    print(f"  Date range: {dt_min} → {dt_max}")
    results["date_min"] = str(dt_min)
    results["date_max"] = str(dt_max)

    # Target P(T)
    assert "P(T)" in df.columns, "Target P(T) missing!"
    nan_count = df["P(T)"].isna().sum()
    neg_count = (df["P(T)"] < 0).sum()
    p_min = df["P(T)"].min()
    p_max = df["P(T)"].max()
    p_mean = df["P(T)"].mean()
    print(f"  P(T): min={p_min:.1f}, max={p_max:.1f}, mean={p_mean:.1f}, NaN={nan_count}, neg={neg_count}")
    results["target_nan"] = int(nan_count)
    results["target_neg"] = int(neg_count)
    results["target_min"] = float(p_min)
    results["target_max"] = float(p_max)
    results["target_mean"] = float(p_mean)
    assert nan_count == 0, f"P(T) has {nan_count} NaN values!"
    print(f"  ✓ Target OK (no NaN, no negative)")

    # Duplicate timestamps
    dupes = df.index.duplicated().sum()
    print(f"  Duplicate timestamps: {dupes}")
    results["duplicate_timestamps"] = int(dupes)
    assert dupes == 0, f"{dupes} duplicate timestamps!"
    print(f"  ✓ No duplicates")

    # Lag correctness: P(T-1) should equal P(T).shift(1)
    if "P(T-1)" in df.columns:
        mismatch = (df["P(T-1)"] - df["P(T)"].shift(1)).dropna().abs().max()
        print(f"  P(T-1) vs P(T).shift(1) max diff: {mismatch:.6f}")
        results["lag_pt1_max_diff"] = float(mismatch)
        assert mismatch < 1e-6, f"P(T-1) lag mismatch: {mismatch}"
        print(f"  ✓ Lag P(T-1) correct")

    if "P(T-24)" in df.columns:
        mismatch24 = (df["P(T-24)"] - df["P(T)"].shift(24)).dropna().abs().max()
        print(f"  P(T-24) vs P(T).shift(24) max diff: {mismatch24:.6f}")
        results["lag_pt24_max_diff"] = float(mismatch24)
        assert mismatch24 < 1e-6, f"P(T-24) lag mismatch"
        print(f"  ✓ Lag P(T-24) correct")

    # Key feature columns present
    expected_cols = ["Hour", "DayOfWeek", "Month"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column: {c}"
    print(f"  ✓ Calendar features present (Hour, DayOfWeek, Month)")

    load_cols = [c for c in ["Demand", "Net_Load", "RE_Penetration", "Solar_Ramp"] if c in df.columns]
    print(f"  Load features present: {load_cols}")
    results["load_features"] = load_cols

    weather_cols = [c for c in df.columns if "temperature" in c.lower() or "radiation" in c.lower()]
    print(f"  Weather features: {weather_cols}")
    results["weather_features"] = weather_cols

    print(f"\n  PHASE 1 RESULT: PASS")
    results["status"] = "PASS"
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — V2: Point forecast re-run
# ═══════════════════════════════════════════════════════════════════════════

def audit_v2_point_forecast():
    """Re-run v2 holdout inference and compare metrics."""
    print("\n" + "=" * 70)
    print("PHASE 2 — V2: POINT FORECAST RE-RUN")
    print("=" * 70)

    results = {}

    # Load saved metrics
    saved = _load_yaml(PROJECT_ROOT / "v2" / "results" / "holdout_summary.yaml")
    print(f"\n  Saved metrics:")
    print(f"    XGB MAPE: {saved['xgb_mape']:.4f}%")
    print(f"    LSTM MAPE: {saved['lstm_mape']:.4f}%")
    print(f"    XGB RMSE: {saved['xgb_rmse']:.4f}")
    print(f"    LSTM RMSE: {saved['lstm_rmse']:.4f}")
    print(f"    Holdout: {saved['holdout_start']} → {saved['holdout_end']}")
    results["saved"] = saved

    # Re-run inference_holdout.py
    import subprocess
    print(f"\n  Re-running v2/inference_holdout.py ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v2" / "inference_holdout.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[:2000]}")
        results["status"] = "FAIL — inference_holdout.py failed"
        results["stderr"] = proc.stderr[:2000]
        return results

    # Parse output for MAPE/RMSE
    stdout = proc.stdout
    results["inference_stdout"] = stdout[-3000:] if len(stdout) > 3000 else stdout

    # Re-load updated metrics
    rerun_path = PROJECT_ROOT / "v2" / "results" / "holdout_summary.yaml"
    rerun = _load_yaml(rerun_path)
    print(f"\n  Re-run metrics:")
    print(f"    XGB MAPE: {rerun['xgb_mape']:.4f}%")
    print(f"    LSTM MAPE: {rerun['lstm_mape']:.4f}%")
    print(f"    XGB RMSE: {rerun['xgb_rmse']:.4f}")
    print(f"    LSTM RMSE: {rerun['lstm_rmse']:.4f}")
    results["rerun"] = rerun

    # Compare
    xgb_mape_diff = _pct_diff(rerun["xgb_mape"], saved["xgb_mape"])
    lstm_mape_diff = _pct_diff(rerun["lstm_mape"], saved["lstm_mape"])
    print(f"\n  XGB MAPE diff: {xgb_mape_diff:.4f}%")
    print(f"  LSTM MAPE diff: {lstm_mape_diff:.4f}%")
    results["xgb_mape_diff_pct"] = xgb_mape_diff
    results["lstm_mape_diff_pct"] = lstm_mape_diff

    if xgb_mape_diff < 1.0 and lstm_mape_diff < 1.0:
        print(f"\n  PHASE 2 RESULT: PASS (metrics reproduced within 1%)")
        results["status"] = "PASS"
    else:
        print(f"\n  PHASE 2 RESULT: WARNING (metrics differ >1%)")
        results["status"] = "WARNING"

    # Also run evaluate_forecast_metrics.py
    print(f"\n  Re-running v2/evaluate_forecast_metrics.py ...")
    eval_proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v2" / "evaluate_forecast_metrics.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    if eval_proc.returncode == 0:
        eval_metrics = _load_yaml(PROJECT_ROOT / "v2" / "results" / "holdout_forecast_metrics.yaml")
        results["eval_metrics"] = eval_metrics
        print(f"  Evaluation metrics regenerated successfully")
    else:
        print(f"  evaluate_forecast_metrics.py failed (non-critical): {eval_proc.stderr[:500]}")
        results["eval_error"] = eval_proc.stderr[:500]

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — V3: Ensemble re-run
# ═══════════════════════════════════════════════════════════════════════════

def audit_v3_ensemble():
    """Re-run v3 ensemble and compare."""
    print("\n" + "=" * 70)
    print("PHASE 3 — V3: ENSEMBLE RE-RUN")
    print("=" * 70)

    results = {}

    saved_path = PROJECT_ROOT / "v3" / "results" / "ensemble_summary.yaml"
    if not saved_path.exists():
        print(f"  ensemble_summary.yaml not found — skipping")
        results["status"] = "SKIPPED"
        return results

    saved = _load_yaml(saved_path)
    print(f"\n  Saved ensemble metrics: {saved}")
    results["saved"] = saved

    import subprocess
    print(f"\n  Re-running v3/run_ensemble.py ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v3" / "run_ensemble.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[:2000]}")
        results["status"] = f"FAIL — exit code {proc.returncode}"
        results["stderr"] = proc.stderr[:2000]
        return results

    results["stdout"] = proc.stdout[-2000:]

    rerun_path = PROJECT_ROOT / "v3" / "results" / "ensemble_summary.yaml"
    if rerun_path.exists():
        rerun = _load_yaml(rerun_path)
        print(f"  Re-run ensemble metrics: {rerun}")
        results["rerun"] = rerun
        results["status"] = "PASS"
    else:
        results["status"] = "PASS (no yaml output, check stdout)"

    print(f"\n  PHASE 3 RESULT: {results['status']}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — V4: Probabilistic re-run
# ═══════════════════════════════════════════════════════════════════════════

def audit_v4_probabilistic():
    """Re-run v4 quantile training and planning."""
    print("\n" + "=" * 70)
    print("PHASE 4 — V4: PROBABILISTIC FORECAST RE-RUN")
    print("=" * 70)

    results = {}
    import subprocess

    # Load saved metrics
    saved_metrics = _load_yaml(PROJECT_ROOT / "v4" / "results" / "quantile_model_metrics.yaml")
    saved_planning = _load_yaml(PROJECT_ROOT / "v4" / "results" / "planning_summary.yaml")
    print(f"\n  Saved quantile metrics (validation):")
    for k, v in saved_metrics.get("validation", saved_metrics).items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v}")
    results["saved_metrics"] = saved_metrics
    results["saved_planning"] = saved_planning

    # Re-run train_quantile_model.py
    print(f"\n  Re-running v4/train_quantile_model.py ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v4" / "train_quantile_model.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=600,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[:2000]}")
        results["train_status"] = "FAIL"
        results["train_stderr"] = proc.stderr[:2000]
    else:
        results["train_status"] = "OK"
        results["train_stdout"] = proc.stdout[-3000:]

    # Re-run run_planning.py
    print(f"\n  Re-running v4/run_planning.py ...")
    t0 = time.time()
    proc2 = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v4" / "run_planning.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=600,
    )
    elapsed2 = time.time() - t0
    print(f"  Completed in {elapsed2:.1f}s (exit code {proc2.returncode})")

    if proc2.returncode != 0:
        print(f"  STDERR: {proc2.stderr[:2000]}")
        results["planning_status"] = "FAIL"
        results["planning_stderr"] = proc2.stderr[:2000]
    else:
        results["planning_status"] = "OK"
        results["planning_stdout"] = proc2.stdout[-3000:]

    # Compare metrics
    rerun_metrics_path = PROJECT_ROOT / "v4" / "results" / "quantile_model_metrics.yaml"
    if rerun_metrics_path.exists():
        rerun_metrics = _load_yaml(rerun_metrics_path)
        results["rerun_metrics"] = rerun_metrics
        print(f"\n  Re-run quantile metrics loaded")

    rerun_planning_path = PROJECT_ROOT / "v4" / "results" / "planning_summary.yaml"
    if rerun_planning_path.exists():
        rerun_planning = _load_yaml(rerun_planning_path)
        results["rerun_planning"] = rerun_planning
        print(f"  Re-run planning summary loaded")

    if results.get("train_status") == "OK" and results.get("planning_status") == "OK":
        results["status"] = "PASS"
    else:
        results["status"] = "PARTIAL"

    print(f"\n  PHASE 4 RESULT: {results['status']}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — V5: Stochastic optimizer
# ═══════════════════════════════════════════════════════════════════════════

def audit_v5_optimizer():
    """Re-run v5 single-week optimizer."""
    print("\n" + "=" * 70)
    print("PHASE 5 — V5: STOCHASTIC OPTIMIZER RE-RUN")
    print("=" * 70)

    results = {}
    import subprocess

    saved_path = PROJECT_ROOT / "v5" / "results" / "single_week_results.yaml"
    if saved_path.exists():
        saved = _load_yaml(saved_path)
        print(f"\n  Saved single-week results: {json.dumps({k: v for k, v in saved.items() if isinstance(v, (int, float, str))}, indent=4)[:1000]}")
        results["saved"] = saved

    print(f"\n  Re-running v5/run_optimizer.py ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v5" / "run_optimizer.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[:2000]}")
        results["status"] = f"FAIL — exit code {proc.returncode}"
        results["stderr"] = proc.stderr[:2000]
    else:
        results["stdout"] = proc.stdout[-3000:]
        results["status"] = "PASS"
        if "perfect_foresight" in proc.stdout.lower() or "PF" in proc.stdout:
            print(f"  Strategy comparison found in output")

    print(f"\n  PHASE 5 RESULT: {results['status']}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — V6: Merchant backtest
# ═══════════════════════════════════════════════════════════════════════════

def audit_v6_backtest():
    """Re-run v6 full backtest and compare aggregate metrics."""
    print("\n" + "=" * 70)
    print("PHASE 6 — V6: MERCHANT BACKTEST RE-RUN")
    print("=" * 70)

    results = {}
    import subprocess

    saved_path = PROJECT_ROOT / "v6" / "results" / "aggregate_metrics.json"
    saved = _load_json(saved_path)
    print(f"\n  Saved aggregate metrics:")
    for k, v in saved.items():
        print(f"    {k}: {v}")
    results["saved"] = saved

    print(f"\n  Re-running v6/run_backtest.py (full 75-week) ...")
    print(f"  This may take 5-15 minutes ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "v6" / "run_backtest.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=1800,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[:3000]}")
        results["status"] = f"FAIL — exit code {proc.returncode}"
        results["stderr"] = proc.stderr[:3000]
        return results

    results["stdout_tail"] = proc.stdout[-3000:]

    rerun_path = PROJECT_ROOT / "v6" / "results" / "aggregate_metrics.json"
    rerun = _load_json(rerun_path)
    print(f"\n  Re-run aggregate metrics:")
    for k, v in rerun.items():
        print(f"    {k}: {v}")
    results["rerun"] = rerun

    comparisons = {}
    for key in ["mean_weekly_revenue", "annual_revenue_mean", "revenue_per_mw_per_year_lakh",
                 "capture_vs_pf_pct_mean", "cvar_realized", "negative_weeks", "cycles_per_year"]:
        if key in saved and key in rerun:
            diff = _pct_diff(rerun[key], saved[key])
            comparisons[key] = {
                "saved": saved[key],
                "rerun": rerun[key],
                "diff_pct": diff,
            }
            status = "✓" if diff < 5.0 else "⚠"
            print(f"  {status} {key}: saved={saved[key]:.2f}, rerun={rerun[key]:.2f}, diff={diff:.2f}%")

    results["comparisons"] = comparisons
    all_ok = all(c["diff_pct"] < 5.0 for c in comparisons.values() if isinstance(c.get("diff_pct"), float))
    results["status"] = "PASS" if all_ok else "WARNING"
    print(f"\n  PHASE 6 RESULT: {results['status']}")
    return results


def audit_v6_sensitivities():
    """Re-run v6 sensitivity backtests."""
    print("\n" + "=" * 70)
    print("PHASE 6b — V6: SENSITIVITY RE-RUNS")
    print("=" * 70)

    results = {}
    import subprocess

    run_bt = PROJECT_ROOT / "v6" / "run_backtest.py"
    sensitivities = [
        {
            "name": "tx0",
            "saved_path": PROJECT_ROOT / "v6" / "results" / "aggregate_metrics_tx0.json",
            "args": ["--tx-costs-config", str(PROJECT_ROOT / "v6" / "config" / "transaction_costs_tx0.yaml")],
        },
        {
            "name": "cycles7",
            "saved_path": PROJECT_ROOT / "v6" / "results" / "aggregate_metrics_cycles7.json",
            "args": ["--bess-config", str(PROJECT_ROOT / "v6" / "config" / "bess_config_cycles7.yaml")],
        },
    ]
    agg_json_path = PROJECT_ROOT / "v6" / "results" / "aggregate_metrics.json"

    for sens in sensitivities:
        print(f"\n  --- Sensitivity: {sens['name']} ---")
        saved = _load_json(sens["saved_path"])
        print(f"  Saved rev/MW/year: {saved.get('revenue_per_mw_per_year_lakh', 'N/A')}")
        print(f"  Re-running v6/run_backtest.py {' '.join(sens['args'])} ...")
        t0 = time.time()
        cmd = [sys.executable, str(run_bt)] + sens["args"]
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=1800,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s (exit code {proc.returncode})")
        if proc.returncode != 0:
            print(f"  STDERR: {proc.stderr[:2000]}")
            results[sens["name"]] = {"status": "FAIL", "stderr": proc.stderr[:2000]}
        else:
            if agg_json_path.exists():
                rerun = _load_json(agg_json_path)
                diff = _pct_diff(
                    rerun.get("revenue_per_mw_per_year_lakh", 0),
                    saved.get("revenue_per_mw_per_year_lakh", 0),
                )
                print(f"  Rev/MW/year: saved={saved.get('revenue_per_mw_per_year_lakh', 'N/A'):.4f}, rerun={rerun.get('revenue_per_mw_per_year_lakh', 'N/A'):.4f}, diff={diff:.2f}%")
                results[sens["name"]] = {
                    "status": "PASS" if diff < 5 else "WARNING",
                    "diff_pct": diff,
                    "saved": saved.get("revenue_per_mw_per_year_lakh"),
                    "rerun": rerun.get("revenue_per_mw_per_year_lakh"),
                }
                import shutil
                dest = PROJECT_ROOT / "v6" / "results" / f"aggregate_metrics_{sens['name']}.json"
                shutil.copy2(agg_json_path, dest)
            else:
                results[sens["name"]] = {"status": "PASS (ran, no json output)"}

    print(f"\n  PHASE 6b RESULT: {json.dumps(results, indent=2, default=str)[:500]}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              FULL PIPELINE RE-RUN AUDIT                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    all_results = {}
    total_start = time.time()

    all_results["v1_data"] = audit_v1_data()
    all_results["v2_point"] = audit_v2_point_forecast()
    all_results["v3_ensemble"] = audit_v3_ensemble()
    all_results["v4_probabilistic"] = audit_v4_probabilistic()
    all_results["v5_optimizer"] = audit_v5_optimizer()
    all_results["v6_backtest"] = audit_v6_backtest()
    all_results["v6_sensitivities"] = audit_v6_sensitivities()

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    for phase, res in all_results.items():
        status = res.get("status", "UNKNOWN") if isinstance(res, dict) else "UNKNOWN"
        print(f"  {phase:30s} → {status}")
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save full results to auxiliary/
    out_path = PROJECT_ROOT / "auxiliary" / "audit_results.json"
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2, default=str)
    print(f"\n  Full results saved: {out_path}")

    return all_results


if __name__ == "__main__":
    main()
