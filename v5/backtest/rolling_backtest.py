"""
Rolling weekly backtest harness for BESS arbitrage strategies.

For each week W in [start_date, end_date]:
    1. FREEZE: knowledge boundary = W_start - 1 hour
    2. SPLIT:  training data = everything before boundary
    3. FORECAST: run v4 quantile model on week W's features
       -> apply conformal calibration
       -> generate 200 scenarios via copula
       -> reduce to 10 scenarios
    4. OPTIMIZE:
       a. stochastic_cvar(scenarios, weights, bess_params)
       b. deterministic_lp(actual_prices)     # perfect foresight
       c. naive_threshold(actual_prices)      # simple baseline
       d. consensus_policy(q_lower, q_upper)  # Columbia approach
    5. SIMULATE: apply each schedule to actual realised prices
    6. RECORD: weekly revenue, cycles, SoC trajectory, CVaR realised

The v4 quantile model is trained ONCE (on data up to Dec 2024) and used
for all backtest weeks. The conformal calibrator and correlation matrix are
re-estimated at each backtest week using a trailing window — this is
realistic (you would recalibrate weekly in production).
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure v4 is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "v4"), str(_PROJECT_ROOT / "v5")]:
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

from v5.optimizer.bess_params import BESSParams
from v5.optimizer.deterministic_lp import solve_deterministic_lp, simulate_schedule
from v5.optimizer.stochastic_cvar import solve_stochastic_cvar
from v5.optimizer.consensus_policy import run_consensus_rolling
from v5.backtest.baselines import naive_threshold_schedule
from v5.backtest.metrics import WeeklyMetrics, compute_weekly_metrics


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WeekResult:
    """Results for a single backtest week."""

    week_start: pd.Timestamp
    week_end: pd.Timestamp
    actual_prices: np.ndarray
    metrics: Dict[str, WeeklyMetrics]        # strategy_name -> WeeklyMetrics
    schedules: Dict[str, Dict[str, np.ndarray]]  # strategy_name -> {p_ch, p_dis, soc}


@dataclass
class BacktestResults:
    """Full backtest results across all weeks."""

    weeks: List[WeekResult]
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Rolling backtest engine
# ---------------------------------------------------------------------------

class RollingBacktest:
    """
    Rolling weekly backtest engine.

    Parameters
    ----------
    planning_df   : The full planning dataset with DatetimeIndex and target column.
    feature_cols  : List of feature column names for the quantile model.
    target_col    : Target column name (default 'P(T)').
    quantile_model: Trained QuantileForecaster (loaded from v4).
    quantiles     : Array of quantile levels the model predicts.
    bess          : BESSParams instance.
    config        : Full optimizer config dict.
    """

    def __init__(
        self,
        planning_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        quantile_model: QuantileForecaster,
        quantiles: np.ndarray,
        bess: BESSParams,
        config: Dict[str, Any],
    ):
        self.df = planning_df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "Timestamp" in self.df.columns:
                self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])
                self.df = self.df.set_index("Timestamp")
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Timestamp' column")

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = quantile_model
        self.quantiles = quantiles
        self.bess = bess
        self.bp = bess.as_dict()
        self.cfg = config

        # Optimisation config
        opt = config.get("optimization", {})
        self.beta = opt.get("beta_risk_aversion", 0.3)
        self.alpha = opt.get("alpha_cvar", 0.95)
        self.solver = opt.get("solver", "highs")
        self.horizon = opt.get("horizon_hours", 168)

        # Consensus config
        cons = config.get("consensus", {})
        self.n_samples = cons.get("n_samples", 100)
        self.threshold = cons.get("action_threshold", 1.0)

        # Naive config
        naive = config.get("naive_thresholds", {})
        self.charge_below = naive.get("charge_below_inr", 3000)
        self.discharge_above = naive.get("discharge_above_inr", 6000)

        # Scenario config
        scen = config.get("scenarios", {}) if "scenarios" in config else {}
        self.n_raw = scen.get("n_raw", 200)
        self.n_reduced = scen.get("n_reduced", 10)
        self.corr_window_weeks = scen.get("correlation_window_weeks", 52)
        self.scen_seed = config.get("random_state", 42)

        # Conformal config
        conf = config.get("conformal", {}) if "conformal" in config else {}
        self.cal_weeks = conf.get("calibration_weeks", 4)
        self.target_coverage = conf.get("target_coverage", 0.90)
        self.conf_lr = conf.get("learning_rate", 0.05)

    def _get_week_boundaries(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """Return list of Monday timestamps for each backtest week."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        # Align to Monday
        if start.weekday() != 0:
            start = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
        weeks = []
        current = start
        while current + pd.Timedelta(hours=self.horizon - 1) <= end:
            # Check we have enough data
            week_end = current + pd.Timedelta(hours=self.horizon - 1)
            if week_end <= self.df.index.max():
                weeks.append(current)
            current += pd.Timedelta(weeks=1)
        return weeks

    def _build_calibration_set(self, boundary: pd.Timestamp) -> tuple:
        """
        Build conformal calibration set using cal_weeks of data
        just before the boundary.
        """
        cal_hours = self.cal_weeks * 168
        cal_end = boundary
        cal_start = cal_end - pd.Timedelta(hours=cal_hours)

        cal_data = self.df.loc[cal_start:cal_end].iloc[:-1]  # exclude boundary itself
        if len(cal_data) < 168:
            return None, None, None

        X_cal = cal_data[self.feature_cols].values
        y_cal = cal_data[self.target_col].values
        q_cal = self.model.predict(X_cal)  # (n, n_quantiles)

        return y_cal, q_cal, cal_data

    def _build_correlation_matrix(self, boundary: pd.Timestamp) -> np.ndarray:
        """Estimate rank correlation from trailing residuals."""
        # Use data up to boundary
        hist = self.df.loc[:boundary].iloc[:-1]
        y_hist = hist[self.target_col].values
        X_hist = hist[self.feature_cols].values

        # Predict median (quantile index for 0.50)
        q_preds = self.model.predict(X_hist)
        median_idx = np.argmin(np.abs(self.quantiles - 0.50))
        y_median = q_preds[:, median_idx]

        residuals = build_weekly_residuals(y_hist, y_median, hours_per_week=168)

        # Use only the last corr_window_weeks
        if len(residuals) > self.corr_window_weeks:
            residuals = residuals[-self.corr_window_weeks:]

        if len(residuals) < 4:
            # Fallback: identity matrix
            return np.eye(self.horizon)

        return estimate_rank_correlation(residuals)

    def run_single_week(self, week_start: pd.Timestamp) -> Optional[WeekResult]:
        """Run all strategies for a single week."""
        week_end = week_start + pd.Timedelta(hours=self.horizon - 1)
        boundary = week_start  # knowledge boundary

        # --- Extract week data ---
        week_data = self.df.loc[week_start:week_end]
        if len(week_data) < self.horizon:
            logger.warning(f"Insufficient data for week {week_start}: {len(week_data)} hours")
            return None

        week_data = week_data.iloc[:self.horizon]
        X_week = week_data[self.feature_cols].values
        actual_prices = week_data[self.target_col].values

        # --- Quantile forecast ---
        q_forecasts = self.model.predict(X_week)  # (168, n_quantiles)

        # --- Conformal calibration ---
        y_cal, q_cal, _ = self._build_calibration_set(boundary)
        if y_cal is not None and len(y_cal) >= 168:
            calibrator = ConformalCalibrator(
                quantiles=self.quantiles.tolist(),
                target_coverage=self.target_coverage,
                learning_rate=self.conf_lr,
            )
            calibrator.fit(y_cal, q_cal)
            q_adjusted = calibrator.adjust(q_forecasts)
        else:
            q_adjusted = q_forecasts

        # --- Build correlation matrix ---
        corr_matrix = self._build_correlation_matrix(boundary)

        # --- Generate scenarios ---
        raw_scenarios = generate_scenarios(
            q_forecasts=q_adjusted,
            quantiles=self.quantiles,
            corr_matrix=corr_matrix,
            n_scenarios=self.n_raw,
            seed=self.scen_seed,
            price_floor=0.0,
        )

        # --- Reduce scenarios ---
        scenarios, weights = forward_reduction(raw_scenarios, n_keep=self.n_reduced)

        # --- Strategies ---
        results_metrics = {}
        results_schedules = {}

        # 1. Perfect foresight
        pf = solve_deterministic_lp(actual_prices, self.bp, self.solver)
        pf_revenue = pf.revenue
        results_schedules["perfect_foresight"] = {
            "p_ch": pf.p_ch, "p_dis": pf.p_dis, "soc": pf.soc,
        }
        results_metrics["perfect_foresight"] = compute_weekly_metrics(
            pf.p_ch, pf.p_dis, pf.soc, actual_prices, pf_revenue, self.bp,
        )

        # 2. Stochastic CVaR
        stoch = solve_stochastic_cvar(
            scenarios, weights, self.bp, self.beta, self.alpha, self.solver,
        )
        results_schedules["stochastic_cvar"] = {
            "p_ch": stoch.p_ch, "p_dis": stoch.p_dis, "soc": stoch.soc,
        }
        results_metrics["stochastic_cvar"] = compute_weekly_metrics(
            stoch.p_ch, stoch.p_dis, stoch.soc, actual_prices, pf_revenue, self.bp,
        )

        # 3. Risk-neutral (beta=0)
        rn = solve_stochastic_cvar(
            scenarios, weights, self.bp, beta=0.0, alpha=self.alpha, solver_name=self.solver,
        )
        results_schedules["risk_neutral"] = {
            "p_ch": rn.p_ch, "p_dis": rn.p_dis, "soc": rn.soc,
        }
        results_metrics["risk_neutral"] = compute_weekly_metrics(
            rn.p_ch, rn.p_dis, rn.soc, actual_prices, pf_revenue, self.bp,
        )

        # 4. Naive threshold
        naive = naive_threshold_schedule(
            actual_prices, self.bp, self.charge_below, self.discharge_above,
        )
        results_schedules["naive_threshold"] = naive
        results_metrics["naive_threshold"] = compute_weekly_metrics(
            naive["p_ch"], naive["p_dis"], naive["soc"],
            actual_prices, pf_revenue, self.bp,
        )

        # 5. Consensus policy
        # Use adjusted q05 and q95 for conformal bounds
        q_lo_idx = np.argmin(np.abs(self.quantiles - 0.05))
        q_hi_idx = np.argmin(np.abs(self.quantiles - 0.95))
        q_lo = q_adjusted[:, q_lo_idx]
        q_hi = q_adjusted[:, q_hi_idx]

        cons = run_consensus_rolling(
            q_lo, q_hi, self.bp, self.n_samples, self.threshold, self.solver, self.scen_seed,
        )
        results_schedules["consensus"] = cons
        results_metrics["consensus"] = compute_weekly_metrics(
            cons["p_ch"], cons["p_dis"], cons["soc"],
            actual_prices, pf_revenue, self.bp,
        )

        return WeekResult(
            week_start=week_start,
            week_end=week_end,
            actual_prices=actual_prices,
            metrics=results_metrics,
            schedules=results_schedules,
        )

    def run(
        self,
        start_date: str,
        end_date: str,
        max_weeks: Optional[int] = None,
        skip_consensus: bool = False,
    ) -> BacktestResults:
        """
        Run the full rolling backtest.

        Parameters
        ----------
        start_date : First Monday of backtest window
        end_date   : End date of backtest window
        max_weeks  : Optional limit on number of weeks (for debugging)
        skip_consensus : If True, skip the slow consensus policy

        Returns
        -------
        BacktestResults with all weekly results
        """
        week_starts = self._get_week_boundaries(start_date, end_date)
        if max_weeks is not None:
            week_starts = week_starts[:max_weeks]

        logger.info(f"Backtest: {len(week_starts)} weeks from {start_date} to {end_date}")

        all_results: List[WeekResult] = []

        for i, ws in enumerate(week_starts):
            t0 = time.time()
            logger.info(f"[{i+1}/{len(week_starts)}] Week starting {ws.strftime('%Y-%m-%d')}")

            try:
                result = self.run_single_week(ws)
                if result is not None:
                    if skip_consensus and "consensus" in result.metrics:
                        del result.metrics["consensus"]
                        del result.schedules["consensus"]
                    all_results.append(result)
                    stoch_rev = result.metrics.get("stochastic_cvar")
                    pf_rev = result.metrics.get("perfect_foresight")
                    if stoch_rev and pf_rev:
                        logger.info(
                            f"  Stochastic: INR {stoch_rev.revenue:,.0f} "
                            f"({stoch_rev.capture_ratio:.1%} capture) | "
                            f"PF: INR {pf_rev.revenue:,.0f}"
                        )
            except Exception as e:
                logger.error(f"  Error on week {ws}: {e}")
                continue
            finally:
                elapsed = time.time() - t0
                logger.info(f"  Elapsed: {elapsed:.1f}s")

        return BacktestResults(weeks=all_results, config=self.cfg)
