#!/usr/bin/env python3
"""
CLI entry point for the full rolling BESS arbitrage backtest (V6: with transaction costs, DSM, degradation).

Usage:
    python v6/run_backtest.py [--config v6/config/optimizer_config.yaml] \
                              [--weeks 78] [--verbose]

This script:
    1. Loads the v4 planning dataset and trained quantile model.
    2. Loads V6 configs: BESS, market, transaction costs.
    3. Initialises the rolling backtest harness with transaction costs.
    4. Runs all strategies week-by-week.
    5. Generates comparison tables, cumulative P&L plots, boxplots, and heatmaps.
    6. Saves everything to v6/results/ including bid files (15-min discretization).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "v4"), str(PROJECT_ROOT / "v6")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v4.models.quantile_xgb import QuantileForecaster
from optimizer.bess_params import BESSParams, load_bess_config, load_config
from backtest.rolling_backtest import RollingBacktest
from backtest.metrics import compute_aggregate_metrics
from optimizer.bid_discretization import hourly_to_15min_bids


def main():
    parser = argparse.ArgumentParser(description="V6 Rolling BESS Backtest")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "optimizer_config.yaml"),
        help="Path to optimizer config YAML",
    )
    parser.add_argument(
        "--bess-config",
        default=None,
        help="Path to BESS config YAML (default: config/bess_config.yaml)",
    )
    parser.add_argument(
        "--tx-costs-config",
        default=None,
        help="Path to transaction costs config YAML (default: config/transaction_costs.yaml)",
    )
    parser.add_argument("--weeks", type=int, default=None, help="Max weeks to backtest (None = all)")
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
    logger = logging.getLogger("v6.backtest")
    logger.setLevel(logging.INFO)

    # ----------------------------------------------------------------
    # Load configs
    # ----------------------------------------------------------------
    v6_root = Path(__file__).resolve().parent
    cfg = load_config(args.config)
    
    # Load BESS config
    if args.bess_config:
        bess_cfg_path = Path(args.bess_config)
    else:
        bess_cfg_path = v6_root / "config" / "bess_config.yaml"
    bess_cfg = load_bess_config(bess_cfg_path)
    bess = BESSParams.from_config(bess_cfg, config_path=bess_cfg_path)
    
    # Load transaction costs config
    if args.tx_costs_config:
        tx_costs_path = Path(args.tx_costs_config)
    else:
        tx_costs_path = v6_root / "config" / "transaction_costs.yaml"
    with open(tx_costs_path) as f:
        tx_costs_cfg = yaml.safe_load(f)
    transaction_costs_dict = tx_costs_cfg.get("costs", {})
    # Add DSM buffer from optimizer config
    opt_cfg = cfg.get("optimization", {})
    transaction_costs_dict["dsm_cost_buffer_inr_mwh"] = opt_cfg.get("dsm_cost_buffer_inr_mwh", 25.0)

    bt_cfg = cfg.get("backtest", {})
    start_date = bt_cfg.get("start_date", "2024-01-01")
    end_date = bt_cfg.get("end_date", "2025-06-10")

    logger.info(f"Config loaded. BESS: {bess.P_max_mw} MW / {bess.E_cap_mwh} MWh")
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

    logger.info(f"Planning dataset: {len(df)} rows, {len(feature_cols)} features")

    # ----------------------------------------------------------------
    # Load v4 quantile model
    # ----------------------------------------------------------------
    v4_results = PROJECT_ROOT / "v4" / "results"
    logger.info(f"Loading quantile model from {v4_results}")
    quantile_model = QuantileForecaster.load(v4_results)
    
    with open(v4_results / "quantile_meta.json") as f:
        import json
        meta = json.load(f)
    quantiles = np.array(meta["quantiles"])
    logger.info(f"Model quantiles: {quantiles.min():.2f} to {quantiles.max():.2f}")

    # ----------------------------------------------------------------
    # Load v4 config for scenario/conformal settings
    # ----------------------------------------------------------------
    v4_config_path = PROJECT_ROOT / "v4" / "config" / "planning_config.yaml"
    if v4_config_path.exists():
        with open(v4_config_path) as f:
            v4_cfg = yaml.safe_load(f)
        # Merge scenario & conformal settings from v4 config
        if "scenarios" not in cfg:
            cfg["scenarios"] = v4_cfg.get("scenarios", {})
        if "conformal" not in cfg:
            cfg["conformal"] = v4_cfg.get("conformal", {})

    # ----------------------------------------------------------------
    # Initialize backtest engine
    # ----------------------------------------------------------------
    logger.info("Initializing rolling backtest engine...")
    backtest = RollingBacktest(
        planning_df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        quantile_model=quantile_model,
        quantiles=quantiles,
        bess=bess,
        config=cfg,
        transaction_costs_dict=transaction_costs_dict,
    )

    # ----------------------------------------------------------------
    # Run backtest
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting rolling backtest...")
    logger.info("=" * 60)
    t0 = time.time()

    results = backtest.run(
        start_date=start_date,
        end_date=end_date,
        max_weeks=args.weeks,
        skip_consensus=True,  # V6: skip consensus by default
    )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Backtest completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Total weeks: {len(results.weeks)}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results_dir = v6_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Save raw weekly results
    weekly_data = []
    for week_result in results.weeks:
        # Also store PF net revenue for the week so CSV is self-contained
        pf_net = week_result.metrics.get("perfect_foresight")
        pf_revenue_inr = pf_net.net_revenue_inr if pf_net else float("nan")
        for strategy, metrics in week_result.metrics.items():
            weekly_data.append({
                "week_start": week_result.week_start.strftime("%Y-%m-%d"),
                "strategy": strategy,
                "gross_revenue_inr": metrics.gross_revenue_inr,
                "iex_fees_inr": metrics.iex_fees_inr,
                "sldc_rldc_fees_inr": metrics.sldc_rldc_fees_inr,
                "tx_loss_inr": metrics.tx_loss_inr,
                "dsm_cost_inr": metrics.dsm_cost_inr,
                "degradation_cost_inr": metrics.degradation_cost_inr,
                "net_revenue_inr": metrics.net_revenue_inr,
                "pf_revenue_inr": pf_revenue_inr,
                "capture_vs_pf_pct": metrics.capture_vs_pf_pct,
                "capture_vs_naive_pct": metrics.capture_vs_naive_pct,
                "cycles": metrics.cycles,
                "cumulative_cycles_ytd": metrics.cumulative_cycles_ytd,
                "peak_soc": metrics.peak_soc,
                "min_soc": metrics.min_soc,
                "avg_soc_pct": metrics.avg_soc_pct,
            })
    
    weekly_df = pd.DataFrame(weekly_data)
    weekly_csv = results_dir / "weekly_results.csv"
    weekly_df.to_csv(weekly_csv, index=False)
    logger.info(f"Saved weekly metrics to {weekly_csv}")

    # Generate bid files (15-min discretization) for stochastic_cvar strategy
    bids_dir = results_dir / "bids"
    bids_dir.mkdir(exist_ok=True)
    
    for week_result in results.weeks:
        if "stochastic_cvar" in week_result.schedules:
            schedule = week_result.schedules["stochastic_cvar"]
            bids = hourly_to_15min_bids(
                schedule["p_ch"].tolist(),
                schedule["p_dis"].tolist(),
                bess.P_max_mw,
                volume_step_mw=0.1,
            )
            bids_df = pd.DataFrame(bids)
            week_str = week_result.week_start.strftime("%Y%m%d")
            bids_file = bids_dir / f"bids_{week_str}.csv"
            bids_df.to_csv(bids_file, index=False)
    
    logger.info(f"Saved bid files to {bids_dir}")

    # Compute aggregate metrics
    logger.info("Computing aggregate metrics...")
    stoch_results = [w.metrics["stochastic_cvar"] for w in results.weeks if "stochastic_cvar" in w.metrics]
    agg_metrics = compute_aggregate_metrics(stoch_results, power_mw=bess.P_max_mw)
    
    # Save aggregate metrics
    agg_dict = {
        "mean_weekly_revenue": agg_metrics.mean_weekly_revenue,
        "annual_revenue_mean": agg_metrics.annual_revenue_mean,
        "revenue_per_mw_per_year_lakh": agg_metrics.revenue_per_mw_per_year_lakh,
        "capture_vs_pf_pct_mean": agg_metrics.capture_vs_pf_pct_mean,
        "capture_vs_pf_pct_std": agg_metrics.capture_vs_pf_pct_std,
        "mean_cycles_per_week": agg_metrics.mean_cycles_per_week,
        "cycles_per_year": agg_metrics.cycles_per_year,
        "cvar_realized": agg_metrics.cvar_realized,
        "max_drawdown": agg_metrics.max_drawdown,
        "negative_weeks": agg_metrics.negative_weeks,
        "mean_gross_revenue": agg_metrics.mean_gross_revenue,
        "mean_total_costs": agg_metrics.mean_total_costs,
    }
    
    agg_json = results_dir / "aggregate_metrics.json"
    with open(agg_json, "w") as f:
        json.dump(agg_dict, f, indent=2)
    logger.info(f"Saved aggregate metrics to {agg_json}")

    # Print summary
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mean weekly net revenue: INR {agg_metrics.mean_weekly_revenue:,.0f}")
    logger.info(f"Mean weekly gross revenue: INR {agg_metrics.mean_gross_revenue:,.0f}")
    logger.info(f"Mean weekly costs: INR {agg_metrics.mean_total_costs:,.0f}")
    logger.info(f"Annual net revenue (extrapolated): INR {agg_metrics.annual_revenue_mean:,.0f}")
    logger.info(f"Revenue per MW per year: INR {agg_metrics.revenue_per_mw_per_year_lakh:.2f} lakh")
    logger.info(f"Capture vs PF: {agg_metrics.capture_vs_pf_pct_mean:.1f}% +/- {agg_metrics.capture_vs_pf_pct_std:.1f}%")
    logger.info(f"Mean cycles per week: {agg_metrics.mean_cycles_per_week:.2f}")
    logger.info(f"Cycles per year: {agg_metrics.cycles_per_year:.1f}")
    logger.info(f"CVaR (worst 5%): INR {agg_metrics.cvar_realized:,.0f}")
    logger.info(f"Negative weeks: {agg_metrics.negative_weeks}/{agg_metrics.total_weeks}")


if __name__ == "__main__":
    main()
