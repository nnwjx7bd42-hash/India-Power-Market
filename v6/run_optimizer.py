#!/usr/bin/env python3
"""
Single-week optimizer runner for V6 (with transaction costs and bid generation).

Usage:
    python v6/run_optimizer.py [--week 2024-01-01] [--config v6/config/optimizer_config.yaml]

Runs all optimization strategies for a single week and generates:
- Cost breakdown table
- 15-minute bid file
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "v4"), str(PROJECT_ROOT / "v6")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v4.models.quantile_xgb import QuantileForecaster
from v4.models.conformal_wrapper import ConformalCalibrator
from v4.scenarios.copula_generator import (
    build_weekly_residuals,
    estimate_rank_correlation,
    generate_scenarios,
)
from v4.scenarios.scenario_reduction import forward_reduction

from optimizer.bess_params import BESSParams, load_bess_config, load_config
from optimizer.deterministic_lp import solve_deterministic_lp
from optimizer.stochastic_cvar import solve_stochastic_cvar
from optimizer.bid_discretization import hourly_to_15min_bids


def main():
    parser = argparse.ArgumentParser(description="V6 Single-Week Optimizer")
    parser.add_argument(
        "--week",
        default="2024-01-01",
        help="Week start date (Monday)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "optimizer_config.yaml"),
        help="Path to optimizer config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("v6.optimizer")

    # Load configs
    v6_root = Path(__file__).resolve().parent
    cfg = load_config(args.config)
    
    bess_cfg_path = v6_root / "config" / "bess_config.yaml"
    bess_cfg = load_bess_config(bess_cfg_path)
    bess = BESSParams.from_config(bess_cfg, config_path=bess_cfg_path)
    
    tx_costs_path = v6_root / "config" / "transaction_costs.yaml"
    with open(tx_costs_path) as f:
        tx_costs_cfg = yaml.safe_load(f)
    transaction_costs_dict = tx_costs_cfg.get("costs", {})
    opt_cfg = cfg.get("optimization", {})
    transaction_costs_dict["dsm_cost_buffer_inr_mwh"] = opt_cfg.get("dsm_cost_buffer_inr_mwh", 25.0)

    # Load planning data
    v4_data = PROJECT_ROOT / "v4" / "data" / "planning_dataset.parquet"
    df = pd.read_parquet(v4_data)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.set_index("Timestamp")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    target_col = "P(T)"
    feature_cols = [c for c in df.columns if c != target_col]

    # Load model
    v4_results = PROJECT_ROOT / "v4" / "results"
    quantile_model = QuantileForecaster.load(v4_results)
    with open(v4_results / "quantile_meta.json") as f:
        import json
        meta = json.load(f)
    quantiles = np.array(meta["quantiles"])

    # Extract week data
    week_start = pd.Timestamp(args.week)
    week_end = week_start + pd.Timedelta(hours=167)
    week_data = df.loc[week_start:week_end].iloc[:168]
    
    if len(week_data) < 168:
        logger.error(f"Insufficient data for week {week_start}")
        return

    X_week = week_data[feature_cols].values
    actual_prices = week_data[target_col].values

    # Forecast
    q_forecasts = quantile_model.predict(X_week)
    
    # Conformal calibration (simplified: use last 4 weeks)
    cal_end = week_start
    cal_start = cal_end - pd.Timedelta(weeks=4)
    cal_data = df.loc[cal_start:cal_end].iloc[:-1]
    if len(cal_data) >= 168:
        X_cal = cal_data[feature_cols].values
        y_cal = cal_data[target_col].values
        q_cal = quantile_model.predict(X_cal)
        calibrator = ConformalCalibrator(
            quantiles=quantiles.tolist(),
            target_coverage=0.90,
            learning_rate=0.05,
        )
        calibrator.fit(y_cal, q_cal)
        q_adjusted = calibrator.adjust(q_forecasts)
    else:
        q_adjusted = q_forecasts

    # Generate scenarios
    hist = df.loc[:week_start].iloc[:-1]
    y_hist = hist[target_col].values
    X_hist = hist[feature_cols].values
    q_preds = quantile_model.predict(X_hist)
    median_idx = np.argmin(np.abs(quantiles - 0.50))
    y_median = q_preds[:, median_idx]
    residuals = build_weekly_residuals(y_hist, y_median, hours_per_week=168)
    if len(residuals) > 52:
        residuals = residuals[-52:]
    if len(residuals) >= 4:
        corr_matrix = estimate_rank_correlation(residuals)
    else:
        corr_matrix = np.eye(168)

    raw_scenarios = generate_scenarios(
        q_forecasts=q_adjusted,
        quantiles=quantiles,
        corr_matrix=corr_matrix,
        n_scenarios=200,
        seed=42,
        price_floor=0.0,
    )
    scenarios, weights = forward_reduction(raw_scenarios, n_keep=10)

    # Run optimizers
    bp = bess.as_dict()
    
    logger.info("Running optimizers...")
    
    # Perfect foresight
    pf = solve_deterministic_lp(actual_prices, bp, transaction_costs_dict)
    logger.info(f"Perfect foresight: Net revenue = INR {pf.revenue:,.0f}")
    
    # Stochastic CVaR
    stoch = solve_stochastic_cvar(
        scenarios, weights, bp, transaction_costs_dict,
        beta=0.3, alpha=0.95,
    )
    logger.info(f"Stochastic CVaR: Expected net revenue = INR {stoch.expected_revenue:,.0f}")
    
    # Generate bids
    bids = hourly_to_15min_bids(
        stoch.p_ch.tolist(),
        stoch.p_dis.tolist(),
        bess.P_max_mw,
    )
    bids_df = pd.DataFrame(bids)
    
    # Print cost breakdown
    print("\n" + "=" * 60)
    print("COST BREAKDOWN (Stochastic CVaR)")
    print("=" * 60)
    
    # Compute costs
    from optimizer.transaction_costs import compute_total_transaction_costs
    tx_costs = compute_total_transaction_costs(
        stoch.p_dis, stoch.p_ch, actual_prices, 1.0, transaction_costs_dict, bp
    )
    
    total_throughput = float(np.sum(stoch.p_dis + stoch.p_ch))
    degradation_cost = total_throughput * bp["C_deg"] / 1000.0
    
    gross_revenue = float(np.sum(actual_prices * (stoch.p_dis * bp.get("eta_discharge", 0.9220) - stoch.p_ch)))
    net_revenue = stoch.expected_revenue
    
    print(f"Gross revenue:        INR {gross_revenue:,.0f}")
    print(f"  IEX fees:           INR {tx_costs['iex_fees_inr']:,.0f}")
    print(f"  SLDC/RLDC fees:     INR {tx_costs['sldc_rldc_fees_inr']:,.0f}")
    print(f"  Transmission loss:  INR {tx_costs['tx_loss_inr']:,.0f}")
    print(f"  Degradation:        INR {degradation_cost:,.0f}")
    print(f"  DSM buffer:         INR {total_throughput * transaction_costs_dict.get('dsm_cost_buffer_inr_mwh', 25.0) / 1000.0:,.0f}")
    print(f"Net revenue:          INR {net_revenue:,.0f}")
    print(f"Capture ratio:        {net_revenue / pf.revenue:.1%}")
    
    # Save bids
    results_dir = v6_root / "results"
    results_dir.mkdir(exist_ok=True)
    bids_file = results_dir / f"bids_{week_start.strftime('%Y%m%d')}.csv"
    bids_df.to_csv(bids_file, index=False)
    print(f"\nSaved bids to {bids_file}")


if __name__ == "__main__":
    main()
