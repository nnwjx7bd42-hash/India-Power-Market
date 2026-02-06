#!/usr/bin/env python3
"""
CLI entry point for the full rolling BESS arbitrage backtest.

Usage:
    python v5/run_backtest.py [--config v5/config/optimizer_config.yaml] \
                              [--weeks 78] [--skip-consensus] [--verbose]

This script:
    1. Loads the v4 planning dataset and trained quantile model.
    2. Initialises the rolling backtest harness.
    3. Runs all strategies week-by-week.
    4. Generates comparison tables, cumulative P&L plots, boxplots, and heatmaps.
    5. Saves everything to v5/results/.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "v4"), str(PROJECT_ROOT / "v5")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v4.models.quantile_xgb import QuantileForecaster
from v5.optimizer.bess_params import BESSParams, load_config
from v5.backtest.rolling_backtest import RollingBacktest
from v5.backtest.report import generate_full_report
from v5.backtest.metrics import compute_aggregate_metrics


def main():
    parser = argparse.ArgumentParser(description="V5 Rolling BESS Backtest")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "optimizer_config.yaml"),
        help="Path to optimizer config YAML",
    )
    parser.add_argument("--weeks", type=int, default=None, help="Max weeks to backtest (None = all)")
    parser.add_argument("--skip-consensus", action="store_true", help="Skip slow consensus policy")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("v5.backtest")
    logger.setLevel(logging.INFO)

    # ----------------------------------------------------------------
    # Load config
    # ----------------------------------------------------------------
    cfg = load_config(args.config)
    bess = BESSParams.from_config(cfg)

    bt_cfg = cfg.get("backtest", {})
    start_date = bt_cfg.get("start_date", "2024-01-01")
    end_date = bt_cfg.get("end_date", "2025-06-10")

    logger.info(f"Config loaded. BESS: {bess.power_mw} MW / {bess.energy_mwh} MWh")
    logger.info(f"Backtest window: {start_date} -> {end_date}")

    # ----------------------------------------------------------------
    # Load planning dataset
    # ----------------------------------------------------------------
    v4_data = PROJECT_ROOT / "v4" / "data" / "planning_dataset.parquet"
    logger.info(f"Loading planning data from {v4_data}")
    df = pd.read_parquet(v4_data)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.set_index("Timestamp")
        else:
            raise ValueError("Dataset must have DatetimeIndex or 'Timestamp' column")

    # Remove timezone for consistent slicing
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    target_col = "P(T)"
    feature_cols = [c for c in df.columns if c != target_col]

    logger.info(f"Dataset: {len(df)} rows, {len(feature_cols)} features")
    logger.info(f"Date range: {df.index.min()} -> {df.index.max()}")

    # ----------------------------------------------------------------
    # Load v4 quantile model
    # ----------------------------------------------------------------
    v4_results = PROJECT_ROOT / "v4" / "results"
    logger.info(f"Loading quantile model from {v4_results}")
    model = QuantileForecaster.load(v4_results)

    with open(v4_results / "quantile_meta.json") as f:
        meta = json.load(f)
    quantiles = np.array(meta["quantiles"])

    logger.info(f"Quantile model loaded: {len(quantiles)} quantiles")

    # ----------------------------------------------------------------
    # Also load planning config for scenario/conformal settings
    # ----------------------------------------------------------------
    v4_config_path = PROJECT_ROOT / "v4" / "config" / "planning_config.yaml"
    with open(v4_config_path) as f:
        v4_cfg = yaml.safe_load(f)

    # Merge scenario & conformal settings from v4 config into optimizer config
    # (these control how scenarios are generated in the backtest)
    if "scenarios" not in cfg:
        cfg["scenarios"] = v4_cfg.get("scenarios", {})
    if "conformal" not in cfg:
        cfg["conformal"] = v4_cfg.get("conformal", {})

    # ----------------------------------------------------------------
    # Run backtest
    # ----------------------------------------------------------------
    engine = RollingBacktest(
        planning_df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        quantile_model=model,
        quantiles=quantiles,
        bess=bess,
        config=cfg,
    )

    logger.info(f"Starting rolling backtest (max_weeks={args.weeks}, "
                f"skip_consensus={args.skip_consensus})")
    t0 = time.time()

    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        max_weeks=args.weeks,
        skip_consensus=args.skip_consensus,
    )

    elapsed = time.time() - t0
    logger.info(f"Backtest complete: {len(results.weeks)} weeks in {elapsed:.0f}s")

    if not results.weeks:
        logger.error("No backtest weeks completed — check date range and data availability.")
        return

    # ----------------------------------------------------------------
    # Generate report
    # ----------------------------------------------------------------
    logger.info("Generating report...")
    table = generate_full_report(results, power_mw=bess.power_mw, save=True)

    # ----------------------------------------------------------------
    # Save raw results (for later analysis)
    # ----------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    # Weekly summary as CSV
    rows = []
    for w in results.weeks:
        for strat, m in w.metrics.items():
            rows.append({
                "week_start": w.week_start,
                "strategy": strat,
                "revenue": m.revenue,
                "capture_ratio": m.capture_ratio,
                "cycles": m.cycles,
                "peak_soc": m.peak_soc,
                "min_soc": m.min_soc,
            })
    weekly_df = pd.DataFrame(rows)
    weekly_df.to_csv(out_dir / "weekly_results.csv", index=False)
    logger.info(f"Weekly results saved to {out_dir / 'weekly_results.csv'}")

    # Aggregate metrics per strategy as YAML
    strategies = weekly_df["strategy"].unique()
    agg_summary = {}
    for strat in strategies:
        strat_metrics = [
            w.metrics[strat] for w in results.weeks if strat in w.metrics
        ]
        agg = compute_aggregate_metrics(strat_metrics, power_mw=bess.power_mw)
        agg_summary[strat] = {
            "mean_weekly_revenue": float(agg.mean_weekly_revenue),
            "std_weekly_revenue": float(agg.std_weekly_revenue),
            "annual_revenue_mean": float(agg.annual_revenue_mean),
            "cvar_realized": float(agg.cvar_realized),
            "max_drawdown": float(agg.max_drawdown),
            "capture_ratio_mean": float(agg.capture_ratio_mean),
            "cycles_per_year": float(agg.cycles_per_year),
            "revenue_per_mw_per_year_lakh": float(agg.revenue_per_mw_per_year_lakh),
            "negative_weeks": int(agg.negative_weeks),
            "total_weeks": int(agg.total_weeks),
        }

    with open(out_dir / "aggregate_metrics.yaml", "w") as f:
        yaml.dump(agg_summary, f, default_flow_style=False)
    logger.info(f"Aggregate metrics saved to {out_dir / 'aggregate_metrics.yaml'}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\nBacktest completed: {len(results.weeks)} weeks in {elapsed:.1f}s")
    print(f"Results saved to {out_dir}/")
    print("Files: backtest_comparison.csv, weekly_results.csv, aggregate_metrics.yaml")
    print("Plots: cumulative_pnl.png, weekly_revenue_boxplots.png, schedule_heatmap_*.png")


if __name__ == "__main__":
    main()
