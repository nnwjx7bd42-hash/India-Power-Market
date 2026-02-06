#!/usr/bin/env python3
"""
Single-week runner: load v4 scenarios, run all optimisation strategies
on the holdout week, compare against perfect foresight and naive baseline.

Usage:
    python v5/run_optimizer.py [--config v5/config/optimizer_config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure v5 is importable
V5_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V5_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v5.optimizer.bess_params import BESSParams, load_config
from v5.optimizer.deterministic_lp import solve_deterministic_lp, simulate_schedule
from v5.optimizer.stochastic_cvar import solve_stochastic_cvar
from v5.optimizer.consensus_policy import run_consensus_rolling


def _naive_threshold_schedule(
    prices: np.ndarray,
    bess_dict: dict,
    charge_below: float = 3000,
    discharge_above: float = 6000,
) -> dict:
    """Simple rule-based strategy: charge when cheap, discharge when expensive."""
    T = len(prices)
    eta = bess_dict["eta"]
    P_max = bess_dict["P_max"]
    E_min = bess_dict["E_min"]
    E_max = bess_dict["E_max"]
    soc = bess_dict["E_init"]

    p_ch = np.zeros(T)
    p_dis = np.zeros(T)
    soc_arr = np.zeros(T)

    for t in range(T):
        if prices[t] <= charge_below:
            max_ch = min(P_max, (E_max - soc) / eta)
            p_ch[t] = max(0, max_ch)
        elif prices[t] >= discharge_above:
            max_dis = min(P_max, (soc - E_min) * eta)
            p_dis[t] = max(0, max_dis)

        soc = soc + eta * p_ch[t] - p_dis[t] / eta
        soc_arr[t] = soc

    return {"p_ch": p_ch, "p_dis": p_dis, "soc": soc_arr}


def main():
    parser = argparse.ArgumentParser(description="V5 single-week optimizer runner")
    parser.add_argument("--config", default=str(V5_ROOT / "config" / "optimizer_config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    bess = BESSParams.from_config(cfg)
    bp = bess.as_dict()

    opt_cfg = cfg.get("optimization", {})
    beta = opt_cfg.get("beta_risk_aversion", 0.3)
    alpha = opt_cfg.get("alpha_cvar", 0.95)
    solver = opt_cfg.get("solver", "highs")

    cons_cfg = cfg.get("consensus", {})
    n_samples = cons_cfg.get("n_samples", 100)
    threshold = cons_cfg.get("action_threshold", 1.0)

    naive_cfg = cfg.get("naive_thresholds", {})
    charge_below = naive_cfg.get("charge_below_inr", 3000)
    discharge_above = naive_cfg.get("discharge_above_inr", 6000)

    # ----------------------------------------------------------------
    # Load v4 outputs
    # ----------------------------------------------------------------
    v4_results = PROJECT_ROOT / "v4" / "results"
    scenarios = np.load(v4_results / "scenarios_reduced.npy")       # (10, 168)
    weights = np.load(v4_results / "scenario_weights.npy")          # (10,)
    holdout_df = pd.read_csv(v4_results / "holdout_quantile_forecasts.csv")
    actual_prices = holdout_df["actual"].values                     # (168,)

    print(f"Loaded {scenarios.shape[0]} scenarios x {scenarios.shape[1]} hours")
    print(f"Actual prices: {len(actual_prices)} hours, "
          f"range [{actual_prices.min():.0f}, {actual_prices.max():.0f}] INR/MWh\n")

    results = {}

    # ----------------------------------------------------------------
    # 1. Perfect foresight (upper bound)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("1) PERFECT FORESIGHT (deterministic LP with actual prices)")
    pf = solve_deterministic_lp(actual_prices, bp, solver)
    results["perfect_foresight"] = pf.revenue
    print(f"   Revenue: INR {pf.revenue:,.0f}  |  Status: {pf.status}")

    # ----------------------------------------------------------------
    # 2. Stochastic CVaR
    # ----------------------------------------------------------------
    print("\n2) STOCHASTIC CVaR (beta={:.2f}, alpha={:.2f})".format(beta, alpha))
    stoch = solve_stochastic_cvar(scenarios, weights, bp, beta, alpha, solver)
    realized_stoch = simulate_schedule(stoch.p_ch, stoch.p_dis, actual_prices, bp)
    results["stochastic_cvar"] = realized_stoch
    print(f"   Expected revenue (scenario-based): INR {stoch.expected_revenue:,.0f}")
    print(f"   CVaR: INR {stoch.cvar:,.0f}  |  VaR: INR {stoch.var:,.0f}")
    print(f"   Realised revenue (vs actual prices): INR {realized_stoch:,.0f}")
    print(f"   Capture ratio: {realized_stoch / pf.revenue:.1%}" if pf.revenue > 0 else "")

    # ----------------------------------------------------------------
    # 3. Risk-neutral (beta=0)
    # ----------------------------------------------------------------
    print("\n3) RISK-NEUTRAL STOCHASTIC (beta=0)")
    rn = solve_stochastic_cvar(scenarios, weights, bp, beta=0.0, alpha=alpha, solver_name=solver)
    realized_rn = simulate_schedule(rn.p_ch, rn.p_dis, actual_prices, bp)
    results["risk_neutral"] = realized_rn
    print(f"   Expected revenue: INR {rn.expected_revenue:,.0f}")
    print(f"   Realised revenue: INR {realized_rn:,.0f}")
    print(f"   Capture ratio: {realized_rn / pf.revenue:.1%}" if pf.revenue > 0 else "")

    # ----------------------------------------------------------------
    # 4. Naive threshold
    # ----------------------------------------------------------------
    print("\n4) NAIVE THRESHOLD (charge < {:.0f}, discharge > {:.0f})".format(
        charge_below, discharge_above
    ))
    naive = _naive_threshold_schedule(actual_prices, bp, charge_below, discharge_above)
    realized_naive = simulate_schedule(naive["p_ch"], naive["p_dis"], actual_prices, bp)
    results["naive_threshold"] = realized_naive
    print(f"   Realised revenue: INR {realized_naive:,.0f}")
    print(f"   Capture ratio: {realized_naive / pf.revenue:.1%}" if pf.revenue > 0 else "")

    # ----------------------------------------------------------------
    # 5. Columbia consensus
    # ----------------------------------------------------------------
    print(f"\n5) COLUMBIA CONSENSUS (n_samples={n_samples}, threshold={threshold})")
    q_lo = holdout_df["q05"].values
    q_hi = holdout_df["q95"].values
    cons = run_consensus_rolling(q_lo, q_hi, bp, n_samples, threshold, solver, seed=42)
    realized_cons = simulate_schedule(cons["p_ch"], cons["p_dis"], actual_prices, bp)
    results["consensus"] = realized_cons
    print(f"   Realised revenue: INR {realized_cons:,.0f}")
    print(f"   Capture ratio: {realized_cons / pf.revenue:.1%}" if pf.revenue > 0 else "")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON — Holdout Week")
    print("=" * 60)
    print(f"{'Strategy':<25} {'Revenue (INR)':>15} {'Capture Ratio':>15}")
    print("-" * 55)
    pf_rev = max(pf.revenue, 1)
    for name, rev in results.items():
        ratio = rev / pf_rev
        print(f"{name:<25} {rev:>15,.0f} {ratio:>14.1%}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    out_dir = V5_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    summary = {k: float(v) for k, v in results.items()}
    summary["beta"] = beta
    summary["alpha"] = alpha
    with open(out_dir / "single_week_results.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    # Save stochastic schedule
    np.savez(
        out_dir / "stochastic_schedule.npz",
        p_ch=stoch.p_ch, p_dis=stoch.p_dis, soc=stoch.soc,
    )

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
