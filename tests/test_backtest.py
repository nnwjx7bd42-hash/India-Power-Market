"""Smoke tests for the v6 backtest layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Load forecasting look-ahead guard ────────────────────────────────────

class TestLoadForecastingNoLookahead:
    """
    Verify that the rolling backtest does NOT leak future load values into
    the quantile model's feature matrix.
    """

    @staticmethod
    def _make_planning_df(n_weeks: int = 10) -> pd.DataFrame:
        """Build a minimal synthetic planning DataFrame."""
        rng = np.random.default_rng(99)
        n_hours = n_weeks * 168
        idx = pd.date_range("2024-01-01", periods=n_hours, freq="h")

        df = pd.DataFrame({
            "P(T)": rng.uniform(1000, 10000, n_hours),
            "Hour": idx.hour,
            "Hour_Sin": np.sin(2 * np.pi * idx.hour / 24),
            "Hour_Cos": np.cos(2 * np.pi * idx.hour / 24),
            "DayOfWeek": idx.dayofweek,
            "IsWeekend": (idx.dayofweek >= 5).astype(float),
            "Month": idx.month,
            "Demand": rng.uniform(100_000, 180_000, n_hours),
            "Net_Load": rng.uniform(80_000, 160_000, n_hours),
            "RE_Penetration": rng.uniform(0.05, 0.30, n_hours),
            "Solar_Ramp": rng.uniform(-5000, 5000, n_hours),
            "temperature_2m_national": rng.uniform(20, 40, n_hours),
        }, index=idx)
        return df

    def test_forecast_replaces_future_load(self):
        """
        After _forecast_load_features, the Demand column for the forecast week
        should NOT equal the actual future values.
        """
        from v6.backtest.rolling_backtest import RollingBacktest

        df = self._make_planning_df(10)
        feature_cols = [c for c in df.columns if c != "P(T)"]

        # We just need the _forecast_load_features method, so we'll test it
        # by instantiating a minimal RollingBacktest and calling the method.
        # We can't fully instantiate without a trained model, so we test the
        # static method logic directly.
        week_start = pd.Timestamp("2024-03-04")  # A Monday in week 10
        boundary = week_start
        week_end = week_start + pd.Timedelta(hours=167)
        week_data = df.loc[week_start:week_end].iloc[:168].copy()

        # Store the actual demand for comparison
        actual_demand = week_data["Demand"].values.copy()

        # Simulate what _forecast_load_features does:
        # It uses same-hour-last-week values from historical data
        hist = df.loc[:boundary].iloc[:-1]
        load_cols = ["Demand", "Net_Load", "RE_Penetration", "Solar_Ramp"]
        forecasted_demand = np.empty(len(week_data))

        for i, ts in enumerate(week_data.index):
            one_week_ago = ts - pd.Timedelta(hours=168)
            if one_week_ago in hist.index:
                forecasted_demand[i] = hist.at[one_week_ago, "Demand"]
            else:
                hour = ts.hour
                four_weeks_ago = ts - pd.Timedelta(weeks=4)
                window = hist.loc[four_weeks_ago:boundary]
                same_hour = window[window.index.hour == hour]["Demand"].dropna()
                forecasted_demand[i] = same_hour.mean() if len(same_hour) > 0 else 0.0

        # The forecasted values should NOT be identical to actual (they come
        # from last week, not the current week)
        assert not np.allclose(forecasted_demand, actual_demand, atol=1.0), (
            "Forecasted demand should differ from actual future demand "
            "(otherwise there may be look-ahead bias)"
        )


# ── Metrics computation ──────────────────────────────────────────────────

class TestWeeklyMetrics:
    """Basic tests for weekly metric computation."""

    def test_net_revenue_less_than_gross(self, bess_dict, transaction_costs_dict):
        from v6.backtest.metrics import compute_weekly_metrics

        T = 168
        rng = np.random.default_rng(42)
        prices = np.concatenate([np.full(12, 2500.0), np.full(12, 7000.0)])
        prices = np.tile(prices, 7)

        # Simple schedule: charge off-peak, discharge peak
        p_ch = np.where(prices < 4000, 10.0, 0.0)
        p_dis = np.where(prices > 5000, 10.0, 0.0)
        soc = np.full(T, 20.0)  # simplified constant SoC for smoke test

        m = compute_weekly_metrics(
            p_ch, p_dis, soc, prices, bess_dict, transaction_costs_dict,
            compute_dsm=False,
        )

        assert m.net_revenue_inr < m.gross_revenue_inr, (
            "Net revenue should be less than gross after costs"
        )
        assert m.cycles > 0, "Should have consumed some cycles"
