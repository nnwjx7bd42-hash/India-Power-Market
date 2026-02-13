import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from pathlib import Path
import os

# Setup themes
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

def plot_cumulative_revenue(df_costs, output_dir):
    print("Plotting cumulative revenue...")
    plt.figure(figsize=(12, 6))
    df_costs = df_costs.sort_values('target_date')
    df_costs['cum_net_revenue'] = df_costs['net_revenue'].cumsum() / 1e6
    
    dates = pd.to_datetime(df_costs['target_date'])
    plt.plot(dates, df_costs['cum_net_revenue'], linewidth=3, color='#2ECC71', label='Cumulative Net Revenue')
    
    plt.title("Post-Recalibration Cumulative Net Revenue (143 Days)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Cumulative Revenue (₹M)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.fill_between(dates, df_costs['cum_net_revenue'], color='#2ECC71', alpha=0.15)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_revenue.png")
    plt.savefig(output_dir / "cumulative_revenue.svg")
    plt.close()

def plot_daily_distribution(df_costs, output_dir):
    print("Plotting daily distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df_costs['net_revenue'] / 1e3, kde=True, color='#3498DB', bins=30, alpha=0.7)
    
    mean_rev = (df_costs['net_revenue'] / 1e3).mean()
    plt.axvline(mean_rev, color='#E74C3C', linestyle='--', linewidth=2, label=f'Mean: ₹{mean_rev:.1f}K')
    
    plt.title("Daily Realized Net Revenue Distribution", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Daily Net Revenue (₹K)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "daily_revenue_distribution.png")
    plt.savefig(output_dir / "daily_revenue_distribution.svg")
    plt.close()

def plot_expected_vs_realized(df_p3b, output_dir):
    print("Plotting expected vs realized...")
    plt.figure(figsize=(10, 10))
    plt.scatter(df_p3b['expected_revenue'] / 1e3, df_p3b['realized_revenue'] / 1e3, 
                alpha=0.6, color='#E67E22', edgecolors='w', s=60)
    
    max_val = max(df_p3b['expected_revenue'].max(), df_p3b['realized_revenue'].max()) / 1e3
    plt.plot([0, max_val], [0, max_val], color='#2C3E50', linestyle='--', alpha=0.5, label='Ideal Alignment')
    
    plt.title("Prediction Integrity: Expected vs. Realized Daily Revenue", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Expected Revenue (Stochastic Optimizer) [₹K]", fontsize=12)
    plt.ylabel("Realized Revenue (Actual Market Prices) [₹K]", fontsize=12)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "expected_vs_realized.png")
    plt.savefig(output_dir / "expected_vs_realized.svg")
    plt.close()

def plot_efficient_frontier(df_cvar, output_dir):
    print("Plotting efficient frontier...")
    plt.figure(figsize=(12, 7))
    
    # Scale worst day to K
    plt.plot(df_cvar['net_revenue_m'], df_cvar['worst_day_k'], marker='o', 
             linewidth=3, markersize=10, color='#9B59B6', label='Stochastic Efficient Frontier')
    
    for i, row in df_cvar.iterrows():
        plt.annotate(f"λ={row['lambda']}", (row['net_revenue_m'], row['worst_day_k']), 
                     textcoords="offset points", xytext=(0,15), ha='center', fontsize=10, fontweight='bold')
    
    plt.title("Efficient Risk-Return Frontier (CVaR Sweep)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Total Net Revenue (₹M)", fontsize=12)
    plt.ylabel("Performance Floor (Worst-Day Return [₹K])", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "efficient_frontier.png")
    plt.savefig(output_dir / "efficient_frontier.svg")
    plt.close()

def plot_forecast_metrics(eval_json, output_dir):
    print("Plotting forecast metrics (WMAPE and Calibration)...")
    
    # 1. WMAPE by Hour
    plt.figure(figsize=(12, 6))
    dam_diag = eval_json['dam_model']['backtest']['hourly_diagnostics']
    rtm_diag = eval_json['rtm_model']['backtest']['hourly_diagnostics']
    
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

def plot_forecast_fan(output_dir):
    print("Plotting forecast fan for sample day...")
    # This is a sample day: April 10, 2025
    sample_date = "2025-04-10"
    
    try:
        # Load recalibrated dam quantiles
        df_dam = pd.read_parquet("Data/Predictions/dam_quantiles_backtest_recalibrated.parquet")
        df_feat = pd.read_parquet("Data/Features/dam_features_backtest.parquet")
        
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
        
        plt.title(f"DAM Price Forecast Fan vs. Realized: {sample_date}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Price (₹/MWh)", fontsize=12)
        plt.xticks(range(24))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / "forecast_fan_sample_day.png")
        plt.savefig(output_dir / "forecast_fan_sample_day.svg")
        plt.close()
    except Exception as e:
        print(f"Error plotting fan: {e}")

def main():
    print("============================================================")
    print("PHASE 5: REGENERATING BACKTEST VISUALIZATIONS")
    print("============================================================")
    
    output_dir = Path("results/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        df_costs = pd.read_csv("Data/Results/phase3c/backtest_results_with_costs.csv")
        df_p3b = pd.read_csv("Data/Results/phase3b/backtest_results.csv")
        df_cvar = pd.read_csv("Data/Backtest/cvar_sweep/efficient_frontier_recalibrated.csv")
        with open("results/forecast_evaluation.json", 'r') as f:
            eval_json = json.load(f)
            
        plot_cumulative_revenue(df_costs, output_dir)
        plot_daily_distribution(df_costs, output_dir)
        plot_expected_vs_realized(df_p3b, output_dir)
        plot_efficient_frontier(df_cvar, output_dir)
        plot_forecast_metrics(eval_json, output_dir)
        plot_forecast_fan(output_dir)
        
        print("\n✅ All charts regenerated in results/charts/")
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
