import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from pathlib import Path
import os
import traceback

# Setup themes
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

def plot_cumulative_revenue(df_results, output_dir):
    print("Plotting cumulative revenue...")
    plt.figure(figsize=(12, 6))
    df_results = df_results.sort_values('target_date')
    df_results['cum_net_revenue'] = df_results['net_revenue'].cumsum() / 1e6
    
    dates = pd.to_datetime(df_results['target_date'])
    plt.plot(dates, df_results['cum_net_revenue'], linewidth=3, color='#2ECC71', label='Cumulative Net Revenue')
    
    plt.title("Cumulative Net Revenue (Soft Terminal + SoC Chaining)\n143-Day Backtest (Feb–Jun 2025)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Cumulative Revenue (Rs M)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.fill_between(dates, df_results['cum_net_revenue'], color='#2ECC71', alpha=0.15)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_revenue.png")
    plt.savefig(output_dir / "cumulative_revenue.svg")
    plt.close()

def plot_daily_distribution(df_results, output_dir):
    print("Plotting daily distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df_results['net_revenue'] / 1e3, kde=True, color='#3498DB', bins=30, alpha=0.7)
    
    mean_rev = (df_results['net_revenue'] / 1e3).mean()
    plt.axvline(mean_rev, color='#E74C3C', linestyle='--', linewidth=2, label=f'Mean: Rs {mean_rev:.1f}K')
    
    plt.title("Daily Realized Net Revenue Distribution\n(Soft Terminal Baseline)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Daily Net Revenue (Rs K)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "daily_revenue_distribution.png")
    plt.savefig(output_dir / "daily_revenue_distribution.svg")
    plt.close()

def plot_expected_vs_realized(df_results, output_dir):
    print("Plotting expected vs realized...")
    # Filter out potential NaNs
    df = df_results.dropna(subset=['expected_revenue', 'realized_revenue'])
    
    plt.figure(figsize=(10, 10))
    plt.scatter(df['expected_revenue'] / 1e3, df['realized_revenue'] / 1e3, 
                alpha=0.6, color='#E67E22', edgecolors='w', s=60)
    
    max_val = max(df['expected_revenue'].max(), df['realized_revenue'].max()) / 1e3
    # Add a buffer
    max_val *= 1.05
    plt.plot([0, max_val], [0, max_val], color='#2C3E50', linestyle='--', alpha=0.5, label='Ideal Alignment')
    
    plt.title("Expected vs. Realized Daily Revenue\n(Soft Terminal Baseline)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Expected Revenue (Stochastic Optimizer) [Rs K]", fontsize=12)
    plt.ylabel("Realized Revenue (Actual Market Prices) [Rs K]", fontsize=12)
    plt.legend()
    plt.axis('equal')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "expected_vs_realized.png")
    plt.savefig(output_dir / "expected_vs_realized.svg")
    plt.close()

def plot_efficient_frontier(df_cvar, output_dir):
    print("Plotting efficient frontier...")
    plt.figure(figsize=(12, 7))
    
    # Check columns
    required_cols = ['net_revenue_m', 'worst_day_k', 'lambda']
    if not all(col in df_cvar.columns for col in required_cols):
        print("Warning: Missing columns for efficient frontier plot. Skipping.")
        return

    plt.plot(df_cvar['net_revenue_m'], df_cvar['worst_day_k'], marker='o', 
             linewidth=3, markersize=10, color='#9B59B6', label='Stochastic Efficient Frontier')
    
    # Custom offsets to avoid overlap for clustered points (L=0, 0.01, 0.05)
    offsets = {
        0.0: (0, 15),
        0.01: (0, -25),
        0.05: (0, 35),
        0.10: (0, -45)
    }
    
    for i, row in df_cvar.iterrows():
        lam = row['lambda']
        offset = offsets.get(lam, (0, 15))
        plt.annotate(f"λ={lam}", (row['net_revenue_m'], row['worst_day_k']), 
                     textcoords="offset points", xytext=offset, ha='center', fontsize=10, fontweight='bold')
    
    plt.title("Efficient Risk-Return Frontier (CVaR Sweep)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Total Net Revenue (Rs M)", fontsize=12)
    plt.ylabel("Performance Floor (Worst-Day Return [Rs K])", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "efficient_frontier.png")
    plt.savefig(output_dir / "efficient_frontier.svg")
    plt.close()

def plot_forecast_metrics(eval_json, output_dir):
    print("Plotting forecast metrics (WMAPE and Calibration)...")
    
    try:
        dam_diag = eval_json['dam_model']['backtest']['hourly_diagnostics']
        rtm_diag = eval_json['rtm_model']['backtest']['hourly_diagnostics']
        
        # 1. WMAPE by Hour
        plt.figure(figsize=(12, 6))
        hours = sorted([int(h) for h in dam_diag.keys()])
        dam_wmape = [dam_diag[str(h)]['wmape'] for h in hours]
        rtm_wmape = [rtm_diag[str(h)]['wmape'] for h in hours]
        
        plt.plot(hours, dam_wmape, label='DAM (24h lead)', marker='o', linewidth=2, color='#E67E22')
        plt.plot(hours, rtm_wmape, label='RTM (1h rolling)', marker='s', linewidth=2, color='#1ABC9C')
        
        plt.title("Forecast Error (WMAPE) by Hour of Day", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Hour of Delivery", fontsize=12)
        plt.ylabel("WMAPE (%)", fontsize=12)
        plt.xticks(range(24))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "forecast_wmape_by_hour.png")
        plt.savefig(output_dir / "forecast_wmape_by_hour.svg")
        plt.close()

        # 2. Calibration
        plt.figure(figsize=(8, 8))
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        dam_prob = eval_json['dam_model']['backtest']['probabilistic']
        rtm_prob = eval_json['rtm_model']['backtest']['probabilistic']
        
        dam_cov = [dam_prob[f'coverage_q{int(q*100)}'] for q in qs]
        rtm_cov = [rtm_prob[f'coverage_q{int(q*100)}'] for q in qs]
        
        plt.plot(qs, dam_cov, 'o-', label='DAM Calibration', linewidth=2, color='#E67E22')
        plt.plot(qs, rtm_cov, 's-', label='RTM Calibration', linewidth=2, color='#1ABC9C')
        plt.plot([0, 1], [0, 1], color='#2C3E50', linestyle='--', alpha=0.5, label='Ideal Calibration')
        
        plt.title("Probabilistic Calibration (Predicted vs. Actual)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Predicted Quantile Probability Level", fontsize=12)
        plt.ylabel("Observed Frequency (Actuals)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "quantile_calibration.png")
        plt.savefig(output_dir / "quantile_calibration.svg")
        plt.close()
    except KeyError as e:
        print(f"Skipping forecast metrics due to missing key: {e}")

def plot_forecast_fan(output_dir):
    print("Plotting forecast fan for sample day...")
    # This is a sample day: April 10, 2025
    sample_date = "2025-04-10"
    
    try:
        # Load recalibrated dam quantiles if available, else raw
        recal_path = "Data/Predictions/dam_quantiles_backtest_recalibrated.parquet"
        raw_path = "Data/Predictions/dam_quantiles_backtest.parquet"
        feat_path = "Data/Features/dam_features_backtest.parquet"
        
        if Path(recal_path).exists():
            df_dam = pd.read_parquet(recal_path)
            title_suffix = "(Recalibrated)"
        elif Path(raw_path).exists():
            df_dam = pd.read_parquet(raw_path)
            title_suffix = "(Raw)"
        else:
            print("Skipping fan plot: No quantile predictions found.")
            return

        if not Path(feat_path).exists():
            print("Skipping fan plot: No features found.")
            return
            
        df_feat = pd.read_parquet(feat_path)
        
        day_pred = df_dam[df_dam['target_date'] == sample_date].sort_values('target_hour')
        day_actual = df_feat[df_feat['target_date'] == sample_date].sort_values('target_hour')
        
        if day_pred.empty or day_actual.empty:
            print(f"Skipping fan plot: No data for {sample_date}")
            return
            
        plt.figure(figsize=(14, 7))
        hours = day_pred['target_hour']
        actual = day_actual['target_mcp_rs_mwh']
        
        plt.fill_between(hours, day_pred['q10'], day_pred['q90'], color='#E67E22', alpha=0.2, label='10-90% Interval')
        plt.fill_between(hours, day_pred['q25'], day_pred['q75'], color='#E67E22', alpha=0.4, label='25-75% Interval')
        plt.plot(hours, day_pred['q50'], color='#E67E22', linewidth=2.5, label='Median Forecast (q50)')
        plt.plot(hours, actual, color='#2C3E50', linewidth=3, marker='o', markersize=4, label='Realized Actual Prices')
        
        plt.title(f"DAM Price Forecast Fan vs. Realized: {sample_date} {title_suffix}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Price (Rs /MWh)", fontsize=12)
        plt.xticks(range(24))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / "forecast_fan_sample_day.png")
        plt.savefig(output_dir / "forecast_fan_sample_day.svg")
        plt.close()
    except Exception as e:
        print(f"Error plotting fan: {e}")
        traceback.print_exc()

def plot_soc_chaining(output_dir):
    print("Plotting SoC chaining trajectory...")
    daily_dir = Path("results/phase3b/daily")
    records = []
    
    if not daily_dir.exists():
        print(f"Skipping SoC chaining: {daily_dir} not found.")
        return

    # Sort files by date in filename
    files = sorted(daily_dir.glob("result_*.json"))
    
    if not files:
        print("Skipping SoC chaining: No result_*.json files found.")
        return

    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            # Ensure keys exist
            if "soc_initial" in d and "soc_terminal" in d:
                records.append({
                    "date": d.get("date", f.stem.replace("result_", "")), # Fallback to filename if date missing
                    "soc_initial": d["soc_initial"],
                    "soc_terminal": d["soc_terminal"],
                })
        except Exception: 
            print(f"Error reading {f}")
            continue
            
    if not records:
        print("Skipping SoC chaining: No valid records found.")
        return

    df = pd.DataFrame(records).sort_values("date")
    dates = pd.to_datetime(df["date"])

    plt.figure(figsize=(14, 5))
    plt.plot(dates, df["soc_terminal"], lw=2, color="#9B59B6",
            label="Terminal SoC (optimizer choice)", marker=".", ms=3)
    plt.plot(dates, df["soc_initial"], lw=1.5, color="#3498DB",
            alpha=0.6, label="Initial SoC (chained)")
    plt.axhline(100, color="#95A5A6", ls=":", lw=1, label="Midpoint (100 MWh)")
    plt.axhline(20, color="#E74C3C", ls=":", lw=1, label="Floor (20 MWh)")
    plt.axhline(180, color="#E74C3C", ls=":", lw=1, alpha=0.5, label="Cap (180 MWh)")
    plt.title("SoC Chaining Trajectory (Soft Terminal Baseline)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("State of Charge (MWh)", fontsize=12)
    plt.ylim(0, 210)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, ls="--", alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "soc_chaining_trajectory.png", dpi=150)
    plt.close()

def main():
    print("============================================================")
    print("PHASE 5: REGENERATING BACKTEST VISUALIZATIONS")
    print("============================================================")
    
    output_dir = Path("results/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Main Results (Soft Terminal Baseline)
    results_path = "results/phase3b/backtest_results.csv"
    if Path(results_path).exists():
        print(f"Loading results from {results_path}...")
        df_results = pd.read_csv(results_path)
        
        try:
            plot_cumulative_revenue(df_results, output_dir)
        except Exception: traceback.print_exc()
            
        try:
            plot_daily_distribution(df_results, output_dir)
        except Exception: traceback.print_exc()
            
        try:
            plot_expected_vs_realized(df_results, output_dir)
        except Exception: traceback.print_exc()
    else:
        print(f"❌ Critical: {results_path} not found. Skipping revenue charts.")

    # Efficient Frontier (CVaR)
    # Efficient Frontier (CVaR)
    cvar_path = "results/cvar_sweep_summary_recalibrated.json"
    if Path(cvar_path).exists():
        try:
            with open(cvar_path, 'r') as f:
                data = json.load(f)
            df_cvar = pd.DataFrame(data)
            plot_efficient_frontier(df_cvar, output_dir)
        except Exception: traceback.print_exc()
    else:
        print(f"⚠️ {cvar_path} not found. Skipping frontier plot.")
        
    # Forecast Metrics
    eval_path = "results/forecast_evaluation.json"
    if Path(eval_path).exists():
        try:
            with open(eval_path, 'r') as f:
                eval_json = json.load(f)
            plot_forecast_metrics(eval_json, output_dir)
        except Exception: traceback.print_exc()
    else:
        print(f"⚠️ {eval_path} not found. Skipping forecast metrics.")

    # Fan Plot
    try:
        plot_forecast_fan(output_dir)
    except Exception: traceback.print_exc()
    
    # SoC Chaining Plot
    try:
        plot_soc_chaining(output_dir)
    except Exception: traceback.print_exc()
        
    print("\n✅ Chart regeneration complete.")

if __name__ == "__main__":
    main()
