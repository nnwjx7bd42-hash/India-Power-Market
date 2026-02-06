"""
Backtest reporting: comparison tables, cumulative P&L, efficient frontier,
weekly boxplots, and schedule heatmaps.

All plots saved to v5/results/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .metrics import AggregateMetrics, WeeklyMetrics, compute_aggregate_metrics
from .rolling_backtest import BacktestResults


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _ensure_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    bt: BacktestResults,
    power_mw: float = 10.0,
) -> pd.DataFrame:
    """
    Build a summary DataFrame: strategy x {mean weekly rev, std, CVaR, capture, cycles}.
    """
    strategies = set()
    for w in bt.weeks:
        strategies.update(w.metrics.keys())
    strategies = sorted(strategies)

    rows = []
    for strat in strategies:
        weekly_metrics = [w.metrics[strat] for w in bt.weeks if strat in w.metrics]
        if not weekly_metrics:
            continue
        agg = compute_aggregate_metrics(weekly_metrics, power_mw=power_mw)
        rows.append({
            "Strategy": strat,
            "Mean Weekly Rev (INR)": agg.mean_weekly_revenue,
            "Std Weekly Rev": agg.std_weekly_revenue,
            "CVaR (5%) Weekly": agg.cvar_realized,
            "Capture Ratio": agg.capture_ratio_mean,
            "Cycles/Year": agg.cycles_per_year,
            "Negative Weeks": agg.negative_weeks,
            "Annual Rev (INR Lakh/MW)": agg.revenue_per_mw_per_year_lakh,
            "Max Drawdown": agg.max_drawdown,
        })

    return pd.DataFrame(rows).set_index("Strategy")


def print_comparison_table(table: pd.DataFrame) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 100)
    print("BACKTEST STRATEGY COMPARISON")
    print("=" * 100)

    fmt = (
        "{:<22} {:>14} {:>12} {:>14} {:>10} {:>10} {:>8} {:>12}"
    )
    print(fmt.format(
        "Strategy", "Mean Rev(INR)", "Std Rev", "CVaR(5%)",
        "Capture", "Cyc/Yr", "NegWk", "Lakh/MW/Yr",
    ))
    print("-" * 100)

    for strat, row in table.iterrows():
        print(fmt.format(
            strat,
            f"{row['Mean Weekly Rev (INR)']:,.0f}",
            f"{row['Std Weekly Rev']:,.0f}",
            f"{row['CVaR (5%) Weekly']:,.0f}",
            f"{row['Capture Ratio']:.1%}",
            f"{row['Cycles/Year']:.1f}",
            f"{int(row['Negative Weeks'])}",
            f"{row['Annual Rev (INR Lakh/MW)']:.1f}",
        ))
    print("=" * 100)


# ---------------------------------------------------------------------------
# 2. Cumulative P&L plot
# ---------------------------------------------------------------------------

def plot_cumulative_pnl(bt: BacktestResults, save: bool = True) -> None:
    """Time series of cumulative revenue for each strategy."""
    _ensure_dir()

    strategies = set()
    for w in bt.weeks:
        strategies.update(w.metrics.keys())
    strategies = sorted(strategies)

    fig, ax = plt.subplots(figsize=(14, 6))

    for strat in strategies:
        dates = []
        cum_rev = []
        total = 0
        for w in bt.weeks:
            if strat in w.metrics:
                dates.append(w.week_start)
                total += w.metrics[strat].revenue
                cum_rev.append(total)

        if dates:
            ax.plot(dates, np.array(cum_rev) / 1e5, label=strat, linewidth=1.5)

    ax.set_xlabel("Week Start")
    ax.set_ylabel("Cumulative Revenue (INR Lakh)")
    ax.set_title("Cumulative P&L by Strategy")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "cumulative_pnl.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3. Efficient frontier (sweep beta)
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    frontier_data: List[Dict],
    save: bool = True,
) -> None:
    """
    Plot expected revenue vs CVaR for different beta values.

    frontier_data: list of dicts with keys 'beta', 'mean_revenue', 'cvar'
    """
    _ensure_dir()

    betas = [d["beta"] for d in frontier_data]
    means = [d["mean_revenue"] for d in frontier_data]
    cvars = [d["cvar"] for d in frontier_data]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(cvars, means, c=betas, cmap="RdYlGn", s=80, zorder=5)
    ax.plot(cvars, means, "k--", alpha=0.3, zorder=1)

    for i, b in enumerate(betas):
        ax.annotate(f"β={b:.1f}", (cvars[i], means[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("Risk Aversion (β)")
    ax.set_xlabel("CVaR (5%) Weekly Revenue (INR)")
    ax.set_ylabel("Mean Weekly Revenue (INR)")
    ax.set_title("Efficient Frontier: Return vs Risk")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "efficient_frontier.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 4. Weekly revenue boxplots
# ---------------------------------------------------------------------------

def plot_weekly_revenue_boxplots(bt: BacktestResults, save: bool = True) -> None:
    """Distribution of weekly revenue per strategy."""
    _ensure_dir()

    strategies = set()
    for w in bt.weeks:
        strategies.update(w.metrics.keys())
    strategies = sorted(strategies)

    data = {}
    for strat in strategies:
        revs = [w.metrics[strat].revenue for w in bt.weeks if strat in w.metrics]
        if revs:
            data[strat] = np.array(revs) / 1e5  # convert to lakh

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(data.keys())
    bp = ax.boxplot([data[s] for s in labels], labels=labels, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Weekly Revenue (INR Lakh)")
    ax.set_title("Weekly Revenue Distribution by Strategy")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "weekly_revenue_boxplots.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 5. Schedule heatmap (avg charge/discharge by hour of day)
# ---------------------------------------------------------------------------

def plot_schedule_heatmap(bt: BacktestResults, strategy: str = "stochastic_cvar", save: bool = True) -> None:
    """Average charge/discharge pattern across hours of day."""
    _ensure_dir()

    # Collect hourly net power (discharge - charge) across all weeks
    hours_of_day = np.arange(24)
    net_power_by_hour = np.zeros((len(bt.weeks), 24 * 7))

    valid_weeks = 0
    for w in bt.weeks:
        if strategy not in w.schedules:
            continue
        sched = w.schedules[strategy]
        T = min(168, len(sched["p_dis"]))
        net = sched["p_dis"][:T] - sched["p_ch"][:T]
        net_power_by_hour[valid_weeks, :T] = net
        valid_weeks += 1

    if valid_weeks == 0:
        return

    net_power_by_hour = net_power_by_hour[:valid_weeks]

    # Reshape to (valid_weeks * 7, 24) and average by hour of day
    avg_by_hour = np.zeros(24)
    counts = np.zeros(24)
    for w in range(valid_weeks):
        for h in range(168):
            hod = h % 24
            avg_by_hour[hod] += net_power_by_hour[w, h]
            counts[hod] += 1

    avg_by_hour = avg_by_hour / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in avg_by_hour]
    ax.bar(hours_of_day, avg_by_hour, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Net Power MW (+discharge / -charge)")
    ax.set_title(f"Average Hourly Schedule: {strategy}")
    ax.set_xticks(hours_of_day)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / f"schedule_heatmap_{strategy}.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Master report function
# ---------------------------------------------------------------------------

def generate_full_report(
    bt: BacktestResults,
    power_mw: float = 10.0,
    save: bool = True,
) -> pd.DataFrame:
    """Generate all report outputs: table + all plots."""
    _ensure_dir()

    # Comparison table
    table = build_comparison_table(bt, power_mw)
    print_comparison_table(table)

    if save:
        table.to_csv(RESULTS_DIR / "backtest_comparison.csv")

    # Plots
    plot_cumulative_pnl(bt, save=save)
    plot_weekly_revenue_boxplots(bt, save=save)

    # Schedule heatmaps for main strategies
    for strat in ["stochastic_cvar", "perfect_foresight", "naive_threshold"]:
        try:
            plot_schedule_heatmap(bt, strategy=strat, save=save)
        except Exception:
            pass

    return table
