"""v6.backtest — Rolling weekly backtest harness with cost modelling."""

from .metrics import WeeklyMetrics, AggregateMetrics, compute_weekly_metrics, compute_aggregate_metrics
from .baselines import naive_threshold_schedule
from .rolling_backtest import RollingBacktest, WeekResult, BacktestResults

__all__ = [
    "WeeklyMetrics",
    "AggregateMetrics",
    "compute_weekly_metrics",
    "compute_aggregate_metrics",
    "naive_threshold_schedule",
    "RollingBacktest",
    "WeekResult",
    "BacktestResults",
]
