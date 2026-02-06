"""
Rolling weekly backtest harness for BESS arbitrage strategies (V6: with transaction costs, DSM, degradation).

For each week W in [start_date, end_date]:
    1. FREEZE: knowledge boundary = W_start - 1 hour
    2. SPLIT:  training data = everything before boundary
    3. FORECAST: run v4 quantile model on week W's features
       -> apply conformal calibration
       -> generate 200 scenarios via copula
       -> reduce to 10 scenarios
    4. OPTIMIZE:
       a. stochastic_cvar(scenarios, weights, bess_params, transaction_costs)
       b. deterministic_lp(actual_prices, transaction_costs)     # perfect foresight
       c. naive_threshold(actual_prices)      # simple baseline
    5. SIMULATE: apply each schedule to actual realised prices
    6. COMPUTE COSTS: transaction costs, DSM costs (with deviation simulation)
    7. RECORD: weekly revenue, cycles, SoC trajectory, all cost components

The v4 quantile model is trained ONCE (on data up to Dec 2024) and used
for all backtest weeks. The conformal calibrator and correlation matrix are
re-estimated at each backtest week using a trailing window — this is
realistic (you would recalibrate weekly in production).

V6 additions:
- Transaction costs (IEX/SLDC/RLDC fees, tx losses) computed per week
- DSM costs computed with deviation simulation
- Degradation tracking: update BESS params at year boundaries
- Cumulative cycle tracking for annual cap enforcement
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure v4 and v6 are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "v4"), str(_PROJECT_ROOT / "v5"), str(_PROJECT_ROOT / "v6")]:
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

from optimizer.bess_params import BESSParams
from optimizer.deterministic_lp import solve_deterministic_lp
from optimizer.stochastic_cvar import solve_stochastic_cvar
from backtest.baselines import naive_threshold_schedule
from backtest.metrics import WeeklyMetrics, compute_weekly_metrics


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
    Rolling weekly backtest engine (V6: with transaction costs, DSM, degradation).

    Parameters
    ----------
    planning_df   : The full planning dataset with DatetimeIndex and target column.
    feature_cols  : List of feature column names for the quantile model.
    target_col    : Target column name (default 'P(T)').
    quantile_model: Trained QuantileForecaster (loaded from v4).
    quantiles     : Array of quantile levels the model predicts.
    bess          : BESSParams instance.
    config        : Full optimizer config dict.
    transaction_costs_dict : Transaction costs dict (from transaction_costs.yaml).
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
        transaction_costs_dict: Dict[str, float],
    ):
        self.df = planning_df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "Timestamp" in self.df.columns:
                self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])
                self.df = self.df.set_index("Timestamp")
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'Timestamp' column")

        # Remove timezone for consistent slicing
        if self.df.index.tz is not None:
            self.df.index = self.df.index.tz_localize(None)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = quantile_model
        self.quantiles = quantiles
        self.bess = bess
        self.bp = bess.as_dict()
        self.cfg = config
        self.transaction_costs_dict = transaction_costs_dict

        # Optimisation config
        opt = config.get("optimization", {})
        self.beta = opt.get("beta_risk_aversion", 0.3)
        self.alpha = opt.get("alpha_cvar", 0.95)
        self.solver = opt.get("solver", "highs")
        self.horizon = opt.get("horizon_hours", 168)

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
        
        # Degradation tracking
        self.commissioning_date = pd.Timestamp("2024-01-01")  # assume commissioning at backtest start
        self.cumulative_cycles_ytd = 0.0

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

    def _update_bess_for_year(self, week_start: pd.Timestamp) -> BESSParams:
        """
        Update BESS parameters for degradation if we've crossed a year boundary.
        
        Returns updated BESSParams instance.
        """
        years_elapsed = (week_start - self.commissioning_date).days / 365.25
        year_offset = int(years_elapsed)
        
        if year_offset > 0:
            # Degrade BESS
            degraded_bess = self.bess.get_bess_params_for_year(year_offset)
            return degraded_bess
        return self.bess

    # Load features that would not be available at week-ahead planning time.
    # In real operation you only have historical load patterns, not future actuals.
    _LOAD_FEATURES_TO_FORECAST = {"Demand", "Net_Load", "RE_Penetration", "Solar_Ramp"}

    def _forecast_load_features(
        self,
        week_data: pd.DataFrame,
        boundary: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Replace future-actual load features with realistic week-ahead proxies.

        At DAM bid-submission time you do **not** know next week's hourly demand.
        This method substitutes actuals with same-hour-last-week values drawn from
        the historical window before *boundary*, eliminating look-ahead bias.

        Falls back to a 4-week rolling mean by hour-of-day if last-week data is
        unavailable for a given hour.
        """
        load_cols = [c for c in self._LOAD_FEATURES_TO_FORECAST if c in self.feature_cols]
        if not load_cols:
            return week_data  # nothing to replace

        week_data = week_data.copy()
        hist = self.df.loc[:boundary].iloc[:-1]  # everything strictly before boundary

        for col in load_cols:
            if col not in hist.columns:
                continue

            forecasted = np.empty(len(week_data))
            for i, ts in enumerate(week_data.index):
                # Primary: same hour, exactly one week ago
                one_week_ago = ts - pd.Timedelta(hours=168)
                if one_week_ago in hist.index and not np.isnan(hist.at[one_week_ago, col]):
                    forecasted[i] = hist.at[one_week_ago, col]
                else:
                    # Fallback: mean of the same hour-of-day over last 4 weeks
                    hour = ts.hour
                    four_weeks_ago = ts - pd.Timedelta(weeks=4)
                    window = hist.loc[four_weeks_ago:boundary]
                    same_hour = window[window.index.hour == hour][col].dropna()
                    forecasted[i] = same_hour.mean() if len(same_hour) > 0 else 0.0

            week_data[col] = forecasted

        return week_data

    def run_single_week(self, week_start: pd.Timestamp, rng: np.random.Generator | None = None) -> Optional[WeekResult]:
        """Run all strategies for a single week (V6: with transaction costs and DSM)."""
        week_end = week_start + pd.Timedelta(hours=self.horizon - 1)
        boundary = week_start  # knowledge boundary

        # Update BESS for degradation if needed
        current_bess = self._update_bess_for_year(week_start)
        current_bp = current_bess.as_dict()

        # --- Extract week data ---
        week_data = self.df.loc[week_start:week_end]
        if len(week_data) < self.horizon:
            logger.warning(f"Insufficient data for week {week_start}: {len(week_data)} hours")
            return None

        week_data = week_data.iloc[:self.horizon]
        actual_prices = week_data[self.target_col].values

        # --- Replace future-actual load features with historical proxies ---
        # This prevents look-ahead bias: at bid-submission time, future hourly
        # demand is unknown.  We use same-hour-last-week as the best proxy.
        week_data = self._forecast_load_features(week_data, boundary)
        X_week = week_data[self.feature_cols].values

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

        # 1. Perfect foresight (with transaction costs)
        pf = solve_deterministic_lp(actual_prices, current_bp, self.transaction_costs_dict, self.solver)
        results_schedules["perfect_foresight"] = {
            "p_ch": pf.p_ch, "p_dis": pf.p_dis, "soc": pf.soc,
        }
        results_metrics["perfect_foresight"] = compute_weekly_metrics(
            pf.p_ch, pf.p_dis, pf.soc, actual_prices, current_bp,
            self.transaction_costs_dict, compute_dsm=True, rng=rng,
        )

        # 2. Stochastic CVaR
        stoch = solve_stochastic_cvar(
            scenarios, weights, current_bp, self.transaction_costs_dict,
            self.beta, self.alpha, self.solver,
        )
        results_schedules["stochastic_cvar"] = {
            "p_ch": stoch.p_ch, "p_dis": stoch.p_dis, "soc": stoch.soc,
        }
        results_metrics["stochastic_cvar"] = compute_weekly_metrics(
            stoch.p_ch, stoch.p_dis, stoch.soc, actual_prices, current_bp,
            self.transaction_costs_dict, compute_dsm=True, rng=rng,
        )

        # 3. Risk-neutral (beta=0)
        rn = solve_stochastic_cvar(
            scenarios, weights, current_bp, self.transaction_costs_dict,
            beta=0.0, alpha=self.alpha, solver_name=self.solver,
        )
        results_schedules["risk_neutral"] = {
            "p_ch": rn.p_ch, "p_dis": rn.p_dis, "soc": rn.soc,
        }
        results_metrics["risk_neutral"] = compute_weekly_metrics(
            rn.p_ch, rn.p_dis, rn.soc, actual_prices, current_bp,
            self.transaction_costs_dict, compute_dsm=True, rng=rng,
        )

        # 4. Naive threshold
        naive = naive_threshold_schedule(
            actual_prices, current_bp, self.charge_below, self.discharge_above,
        )
        results_schedules["naive_threshold"] = naive
        results_metrics["naive_threshold"] = compute_weekly_metrics(
            naive["p_ch"], naive["p_dis"], naive["soc"],
            actual_prices, current_bp,
            self.transaction_costs_dict, compute_dsm=True, rng=rng,
        )

        # --- Capture ratios (all simulated, same actual price path) ---
        pf_net = results_metrics["perfect_foresight"].net_revenue_inr
        naive_net = results_metrics["naive_threshold"].net_revenue_inr

        for name, m in results_metrics.items():
            m.capture_vs_pf_pct = (
                float(np.clip(100.0 * m.net_revenue_inr / pf_net, -500, 500))
                if pf_net != 0 else float("nan")
            )
            m.capture_vs_naive_pct = (
                float(np.clip(100.0 * m.net_revenue_inr / naive_net, -500, 500))
                if naive_net != 0 else float("nan")
            )
            m.cumulative_cycles_ytd = self.cumulative_cycles_ytd + m.cycles

        # Update cumulative cycles (use stochastic_cvar as reference)
        self.cumulative_cycles_ytd += results_metrics["stochastic_cvar"].cycles

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
        skip_consensus: bool = True,  # V6: skip by default (was slow in v5)
    ) -> BacktestResults:
        """
        Run the full rolling backtest (V6: with transaction costs, DSM, degradation).

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
        rng = np.random.default_rng(self.scen_seed)

        for i, ws in enumerate(week_starts):
            t0 = time.time()
            logger.info(f"[{i+1}/{len(week_starts)}] Week starting {ws.strftime('%Y-%m-%d')}")

            try:
                result = self.run_single_week(ws, rng=rng)
                if result is not None:
                    all_results.append(result)
                    stoch_rev = result.metrics.get("stochastic_cvar")
                    pf_rev = result.metrics.get("perfect_foresight")
                    if stoch_rev and pf_rev:
                        logger.info(
                            f"  Stochastic: Net INR {stoch_rev.net_revenue_inr:,.0f} "
                            f"({stoch_rev.capture_vs_pf_pct:.1f}% vs PF) | "
                            f"PF Net: INR {pf_rev.net_revenue_inr:,.0f}"
                        )
            except Exception as e:
                logger.error(f"  Error on week {ws}: {e}", exc_info=True)
                continue
            finally:
                elapsed = time.time() - t0
                logger.info(f"  Elapsed: {elapsed:.1f}s")

        return BacktestResults(weeks=all_results, config=self.cfg)
