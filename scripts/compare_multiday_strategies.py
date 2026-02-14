"""
Multi-Day Strategy Comparison Dashboard

Loads backtest results from multiple strategies and generates:
1. Comparison table (printed + CSV)
2. Daily revenue overlay chart
3. Cumulative revenue curve
4. SoC trajectory
5. Worst-week analysis
"""
import pandas as pd
import numpy as np
import json
import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams.update({
    'figure.figsize': (14, 6),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150
})

STRATEGIES = {
    'baseline': {
        'label': 'Baseline (Hard Terminal)',
        'results_dir': 'results/phase3b',
        'color': '#7f8c8d',
        'linestyle': '-'
    },
    'soft_terminal': {
        'label': 'Soft Terminal (+SoC Chaining)',
        'results_dir': 'results/phase3b_backtest_soft',
        'color': '#2ecc71',
        'linestyle': '-'
    },
    'rolling_optA': {
        'label': '48h Rolling (Option A)',
        'results_dir': 'results/phase4_rolling',
        'color': '#3498db',
        'linestyle': '--'
    },
    'rolling_optB': {
        'label': '48h Rolling (Option B)',
        'results_dir': 'results/phase4_rolling_optb',
        'color': '#e74c3c',
        'linestyle': '-'
    },
    'multiday_optA': {
        'label': '7-Day Extensive (Option A)',
        'results_dir': 'results/phase4_multiday',
        'color': '#9b59b6',
        'linestyle': '--'
    },
    'multiday_optB': {
        'label': '7-Day Extensive (Option B)',
        'results_dir': 'results/phase4_multiday_optb',
        'color': '#e67e22',
        'linestyle': '-'
    }
}


def load_strategy_data(strategy_name: str, config: dict) -> dict:
    """Load backtest results + summary for a strategy."""
    results_dir = Path(config['results_dir'])
    
    data = {'name': strategy_name, 'label': config['label'], 'exists': False}
    
    # Try summary.json first
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data['summary'] = json.load(f)
        data['exists'] = True
    
    # Try daily results CSV
    csv_path = results_dir / "backtest_results.csv"
    if csv_path.exists():
        data['daily'] = pd.read_csv(csv_path)
        data['exists'] = True
    
    return data


def run_comparison():
    parser = argparse.ArgumentParser(description='Compare Multi-Day Strategies')
    parser.add_argument('--output-dir', type=str, default='results/multiday_comparison')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("============================================================")
    print("MULTI-DAY STRATEGY COMPARISON DASHBOARD")
    print("============================================================\n")
    
    # Load all strategies
    strategies = {}
    for name, config in STRATEGIES.items():
        data = load_strategy_data(name, config)
        if data['exists']:
            strategies[name] = data
            print(f"✓ Loaded: {config['label']}")
        else:
            print(f"✗ Missing: {config['label']} ({config['results_dir']})")
    
    if not strategies:
        print("ERROR: No strategy results found!")
        return
    
    # ─── 1. Comparison Table ───
    print("\n═══════════════════════════════════════════════════════════════")
    print("STRATEGY COMPARISON TABLE")
    print("═══════════════════════════════════════════════════════════════")
    
    table_rows = []
    baseline_net = None
    
    for name, data in strategies.items():
        if 'summary' not in data:
            continue
        s = data['summary']
        total_net = s.get('total_net_revenue', 0) / 1e6
        avg_daily = s.get('avg_daily_net', 0) / 1e3
        std_daily = s.get('std_daily_net', 0) / 1e3
        worst = s.get('min_daily_net', 0) / 1e3
        n_days = s.get('n_days', 0)
        
        if name == 'baseline':
            baseline_net = total_net
        
        uplift = ((total_net / baseline_net - 1) * 100) if baseline_net and baseline_net > 0 else 0
        
        row = {
            'Strategy': data['label'],
            'Days': n_days,
            'Total Net (₹M)': f'{total_net:.2f}',
            'Avg Daily (₹K)': f'{avg_daily:.1f}',
            'Std Daily (₹K)': f'{std_daily:.1f}',
            'Worst Day (₹K)': f'{worst:.1f}',
            'Uplift vs Baseline': f'{uplift:+.1f}%'
        }
        
        # SoC stats if available
        if 'avg_terminal_soc' in s:
            row['Avg SoC₂₄'] = f"{s['avg_terminal_soc']:.0f}"
            row['Min SoC₂₄'] = f"{s['min_terminal_soc']:.0f}"
        
        table_rows.append(row)
    
    table_df = pd.DataFrame(table_rows)
    print(table_df.to_string(index=False))
    table_df.to_csv(output_dir / "comparison_table.csv", index=False)
    
    # ─── 2. Daily Revenue Overlay ───
    fig, ax = plt.subplots(figsize=(16, 6))
    
    for name, data in strategies.items():
        if 'daily' not in data:
            continue
        daily = data['daily']
        config = STRATEGIES[name]
        ax.plot(
            range(len(daily)), daily['net_revenue'] / 1e3,
            label=config['label'], color=config['color'],
            linestyle=config['linestyle'], alpha=0.8, linewidth=1.2
        )
    
    ax.set_xlabel('Backtest Day')
    ax.set_ylabel('Daily Net Revenue (₹ thousands)')
    ax.set_title('Multi-Day Strategy Comparison: Daily Net Revenue')
    ax.legend(loc='upper left', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "daily_revenue_overlay.png")
    plt.close(fig)
    print(f"\n✓ Saved daily_revenue_overlay.png")
    
    # ─── 3. Cumulative Revenue Curve ───
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for name, data in strategies.items():
        if 'daily' not in data:
            continue
        daily = data['daily']
        config = STRATEGIES[name]
        cumrev = daily['net_revenue'].cumsum() / 1e6
        ax.plot(
            range(len(cumrev)), cumrev,
            label=config['label'], color=config['color'],
            linestyle=config['linestyle'], linewidth=1.5
        )
    
    ax.set_xlabel('Backtest Day')
    ax.set_ylabel('Cumulative Net Revenue (₹ million)')
    ax.set_title('Cumulative Revenue Trajectories')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_revenue.png")
    plt.close(fig)
    print(f"✓ Saved cumulative_revenue.png")
    
    # ─── 4. SoC Trajectory ───
    soc_strategies = [n for n, d in strategies.items() if 'daily' in d and 'soc_terminal' in d['daily'].columns]
    
    if soc_strategies:
        fig, ax = plt.subplots(figsize=(14, 5))
        
        for name in soc_strategies:
            daily = strategies[name]['daily']
            config = STRATEGIES[name]
            ax.plot(
                range(len(daily)), daily['soc_terminal'],
                label=config['label'], color=config['color'],
                linestyle=config['linestyle'], linewidth=1.2
            )
        
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, label='Initial SoC (100 MWh)')
        ax.axhline(y=20, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Physical Floor (20 MWh)')
        ax.set_xlabel('Backtest Day')
        ax.set_ylabel('Terminal SoC (MWh)')
        ax.set_title('SoC Trajectory Across Strategies')
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(0, 220)
        fig.tight_layout()
        fig.savefig(output_dir / "soc_trajectory.png")
        plt.close(fig)
        print(f"✓ Saved soc_trajectory.png")
    
    # ─── 5. Worst-Week Analysis ───
    print("\n═══════════════════════════════════════════════════════════════")
    print("WORST WEEK ANALYSIS (7-day rolling sum)")
    print("═══════════════════════════════════════════════════════════════")
    
    for name, data in strategies.items():
        if 'daily' not in data or len(data['daily']) < 7:
            continue
        daily = data['daily']
        rolling_7d = daily['net_revenue'].rolling(7).sum()
        worst_idx = rolling_7d.idxmin()
        if pd.isna(worst_idx):
            continue
        worst_week_rev = rolling_7d.iloc[int(worst_idx)] / 1e6
        worst_week_start = daily.iloc[int(worst_idx) - 6]['target_date'] if worst_idx >= 6 else daily.iloc[0]['target_date']
        worst_week_end = daily.iloc[int(worst_idx)]['target_date']
        
        print(f"  {STRATEGIES[name]['label'][:30]:30s}  ₹{worst_week_rev:.2f}M  ({worst_week_start} → {worst_week_end})")
    
    print(f"\n✓ All outputs saved to {output_dir}")
    print("============================================================")


if __name__ == "__main__":
    run_comparison()
