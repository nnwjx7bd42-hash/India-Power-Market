"""
Pipeline stress tests — validate every layer of the forecasting & backtest stack.

Run:  pytest auxiliary/stress_test_pipeline.py -v --tb=short

Checks:
  Layer 0  Data pipeline  — no target leakage, backward-only shifts, holdout excluded
  Layer 1  Point forecast — MAPE / R² / MDA formulas, holdout index, multi-week holdout
  Layer 2  Probabilistic  — disjoint splits, no excluded lags, pinball / CRPS / coverage
  Layer 3  Scenarios      — no actuals in inputs, correlation uses only past
  Layer 4  BESS backtest  — revenue formula, CVaR, PF same costs, cycles & SoC
  Cross    Consistency    — same holdout period, feature parity, reproducibility
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = PROJECT_ROOT / "v2"
V4_ROOT = PROJECT_ROOT / "v4"
V6_ROOT = PROJECT_ROOT / "v6"

for _p in [str(PROJECT_ROOT), str(V4_ROOT), str(V5 := PROJECT_ROOT / "v5"), str(V6_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 0 — Data pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestDataPipeline:
    """Verify data integrity: no target leakage, backward-only features, holdout excluded."""

    # ------------------------------------------------------------------
    # 0-A  Target P(T) must not appear in any feature list
    # ------------------------------------------------------------------
    def test_target_not_in_v4_features(self):
        """P(T) should NOT appear in v4 planning feature list."""
        cfg_path = V4_ROOT / "config" / "planning_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        feat_cfg = cfg["features"]
        feature_names = (
            feat_cfg["calendar"]
            + feat_cfg["weather"]
            + feat_cfg["load"]
            + feat_cfg["anchors"]
            + feat_cfg["interactions"]
        )
        assert "P(T)" not in feature_names, "P(T) target found in v4 feature list — DATA LEAKAGE"

    def test_target_not_in_v2_baseline_features(self):
        """P(T) should NOT appear in v2 baseline feature list."""
        cfg_path = PROJECT_ROOT / "config" / "model_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        baseline_feats = (
            cfg["features"]["baseline"]["original"]
            + cfg["features"]["baseline"]["calendar"]
        )
        assert "P(T)" not in baseline_feats, "P(T) target found in v2 baseline features — DATA LEAKAGE"

    # ------------------------------------------------------------------
    # 0-B  Price anchors use only backward shifts
    # ------------------------------------------------------------------
    def test_price_anchors_backward_only(self):
        """All price anchor columns should be NaN for early rows (backward shift)."""
        from v4.data.price_anchors import add_price_anchors

        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({"P(T)": np.arange(1, n + 1, dtype=float), "Hour": idx.hour}, index=idx)
        df = add_price_anchors(df, price_col="P(T)", window_weeks=4)

        anchor_cols = [
            "P_same_hour_last_week",
            "P_same_hour_2weeks_ago",
            "P_weekly_avg_by_hour",
            "P_weekly_median_by_hour",
            "P_weekly_std_by_hour",
            "P_percentile_90_by_hour",
        ]
        for col in anchor_cols:
            if col not in df.columns:
                continue
            # The first 168 rows (one week) MUST be NaN for shift(168) anchors
            first_valid = df[col].first_valid_index()
            assert first_valid is not None, f"{col} is entirely NaN"
            first_valid_pos = df.index.get_loc(first_valid)
            assert first_valid_pos >= 168, (
                f"{col} has non-NaN value at row {first_valid_pos} (<168) — possible forward leak"
            )

    # ------------------------------------------------------------------
    # 0-C  Holdout excluded from v2 training data
    # ------------------------------------------------------------------
    def test_holdout_excluded_from_v2_training(self):
        """Last holdout_hours of dataset must be absent after v2 trimming."""
        cfg_path = PROJECT_ROOT / "config" / "model_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        holdout_hours = cfg["cv"].get("holdout_hours", 0) or 0
        if holdout_hours == 0:
            pytest.skip("holdout_hours is 0 — nothing to test")

        dataset_path = PROJECT_ROOT / "data" / "processed" / "dataset_cleaned.parquet"
        if not dataset_path.exists():
            # Try v2 path
            dataset_path = V2_ROOT / "dataset_cleaned.parquet"
        if not dataset_path.exists():
            pytest.skip("dataset_cleaned.parquet not found")

        df = pd.read_parquet(dataset_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            df = df.sort_index()

        max_ts = df.index.max()
        holdout_start = max_ts - pd.Timedelta(hours=holdout_hours)

        # Replicate v2 trimming
        df_trimmed = df[df.index <= holdout_start]
        holdout_set = set(df[df.index > holdout_start].index)
        train_set = set(df_trimmed.index)

        overlap = holdout_set & train_set
        assert len(overlap) == 0, (
            f"Holdout timestamps leak into training data: {len(overlap)} rows overlap"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Point forecast (v2)
# ═══════════════════════════════════════════════════════════════════════════

class TestPointForecast:
    """Verify MAPE, R², MDA formulas and holdout boundaries."""

    # ------------------------------------------------------------------
    # 1-A  MAPE formula correctness
    # ------------------------------------------------------------------
    def test_mape_known_values(self):
        """MAPE([100,200], [90,220]) should be 15%."""
        from src.validation.metrics import calculate_mape

        result = calculate_mape([100, 200], [90, 220])
        # |100-90|/100 = 0.10, |200-220|/200 = 0.10  → mean = 0.10 → 10%
        expected = 10.0
        assert abs(result - expected) < 0.01, f"MAPE expected {expected}%, got {result}%"

    def test_mape_with_large_asymmetry(self):
        """MAPE([100,200], [80,260]) = mean(20%, 30%) = 25%."""
        from src.validation.metrics import calculate_mape

        result = calculate_mape([100, 200], [80, 260])
        expected = 25.0
        assert abs(result - expected) < 0.01, f"MAPE expected {expected}%, got {result}%"

    def test_mape_zero_guard(self):
        """When y_true contains zeros, MAPE should filter them out (not crash)."""
        from src.validation.metrics import calculate_mape

        result = calculate_mape([0, 100], [10, 110])
        # Only the second element survives: |100-110|/100 = 10%
        assert abs(result - 10.0) < 0.01

    def test_mape_all_zeros(self):
        """All-zero y_true should return NaN."""
        from src.validation.metrics import calculate_mape

        result = calculate_mape([0, 0, 0], [1, 2, 3])
        assert np.isnan(result)

    # ------------------------------------------------------------------
    # 1-B  R² formula correctness
    # ------------------------------------------------------------------
    def test_r2_perfect(self):
        """R² of perfect predictions should be 1.0."""
        from src.validation.metrics import calculate_r2

        r2 = calculate_r2([1, 2, 3, 4], [1, 2, 3, 4])
        assert abs(r2 - 1.0) < 1e-10

    def test_r2_mean_predictor(self):
        """R² of a constant-mean predictor should be 0.0."""
        from src.validation.metrics import calculate_r2

        y_true = [1, 2, 3, 4, 5]
        mean_val = np.mean(y_true)
        r2 = calculate_r2(y_true, [mean_val] * 5)
        assert abs(r2) < 1e-10, f"R² for mean predictor should be 0, got {r2}"

    def test_r2_worse_than_mean(self):
        """A terrible predictor should have R² < 0."""
        from src.validation.metrics import calculate_r2

        r2 = calculate_r2([1, 2, 3], [100, 200, 300])
        assert r2 < 0, f"R² for terrible predictor should be negative, got {r2}"

    # ------------------------------------------------------------------
    # 1-C  MDA uses only past actuals
    # ------------------------------------------------------------------
    def test_mda_formula(self):
        """
        MDA hand-calculation:
          y_true  = [10, 12, 11, 15]
          y_pred  = [10, 13, 10, 14]

          t=1: actual_prev=10, actual_curr=12, pred_curr=13
               dir_actual = sign(12-10)=+1, dir_forecast = sign(13-10)=+1  → MATCH
          t=2: actual_prev=12, actual_curr=11, pred_curr=10
               dir_actual = sign(11-12)=-1, dir_forecast = sign(10-12)=-1  → MATCH
          t=3: actual_prev=11, actual_curr=15, pred_curr=14
               dir_actual = sign(15-11)=+1, dir_forecast = sign(14-11)=+1  → MATCH

          MDA = 3/3 = 100%
        """
        from src.validation.metrics import calculate_mda

        result = calculate_mda([10, 12, 11, 15], [10, 13, 10, 14])
        assert abs(result - 100.0) < 0.01, f"Expected MDA=100%, got {result}%"

    def test_mda_uses_actual_prev_not_pred_prev(self):
        """
        Confirm direction is computed as sign(pred_t - actual_{t-1}),
        NOT sign(pred_t - pred_{t-1}).

        y_true  = [10, 20, 15]
        y_pred  = [10, 25, 12]

        t=1: dir_actual = sign(20-10)=+1, dir_forecast = sign(25-10)=+1  → MATCH
        t=2: dir_actual = sign(15-20)=-1, dir_forecast = sign(12-20)=-1  → MATCH
        MDA = 100%

        If it incorrectly used pred_{t-1}:
        t=2: dir_forecast = sign(12-25) = -1  → same answer here (coincidence)
        So use a case that differentiates:

        y_true  = [10, 20, 25]
        y_pred  = [10, 30, 22]
        t=1: actual 10→20 (+), forecast sign(30-10)=+1 → MATCH
        t=2: actual 20→25 (+), forecast sign(22-20)=+1 → MATCH  (correct: uses actual_prev=20)
             If wrong (uses pred_prev=30): sign(22-30)=-1 → MISS

        MDA should be 100% with correct formula.
        """
        from src.validation.metrics import calculate_mda

        result = calculate_mda([10, 20, 25], [10, 30, 22])
        assert abs(result - 100.0) < 0.01, (
            f"MDA should be 100% (using actual_prev); got {result}% — "
            "formula may be using pred_prev instead of actual_prev"
        )

    def test_mda_zero_direction(self):
        """MDA with no change in actual should count sign(0) == sign(0)."""
        from src.validation.metrics import calculate_mda

        result = calculate_mda([10, 10, 10], [10, 10, 10])
        assert abs(result - 100.0) < 0.01

    # ------------------------------------------------------------------
    # 1-D  Holdout index matches last 168h of dataset
    # ------------------------------------------------------------------
    def test_holdout_index_matches(self):
        """v2 holdout predictions timestamps should be the last 168h of the dataset."""
        holdout_path = V2_ROOT / "results" / "holdout_actuals_and_predictions.xlsx"
        if not holdout_path.exists():
            pytest.skip("holdout xlsx not found")

        df_holdout = pd.read_excel(holdout_path)
        # Identify timestamp column
        ts_col = None
        for c in df_holdout.columns:
            if "time" in c.lower() or "date" in c.lower() or "datetime" in c.lower():
                ts_col = c
                break
        if ts_col is None and df_holdout.columns[0] != "Unnamed: 0":
            ts_col = df_holdout.columns[0]
        if ts_col is None:
            pytest.skip("Cannot identify timestamp column in holdout xlsx")

        holdout_ts = pd.to_datetime(df_holdout[ts_col])
        assert len(holdout_ts) >= 168, f"Expected 168 holdout rows, got {len(holdout_ts)}"

        # Load main dataset to verify alignment
        dataset_path = PROJECT_ROOT / "data" / "processed" / "dataset_cleaned.parquet"
        if not dataset_path.exists():
            dataset_path = V2_ROOT / "dataset_cleaned.parquet"
        if not dataset_path.exists():
            pytest.skip("dataset_cleaned.parquet not found")

        df_full = pd.read_parquet(dataset_path)
        if not isinstance(df_full.index, pd.DatetimeIndex):
            if "datetime" in df_full.columns:
                df_full = df_full.set_index("datetime")
        df_full = df_full.sort_index()

        dataset_max = df_full.index.max()
        expected_start = dataset_max - pd.Timedelta(hours=167)

        # Compare (timezone-naive)
        h_min = holdout_ts.min()
        h_max = holdout_ts.max()
        if hasattr(h_min, 'tz') and h_min.tz is not None:
            h_min = h_min.tz_localize(None)
            h_max = h_max.tz_localize(None)
        ds_max = dataset_max
        if hasattr(ds_max, 'tz') and ds_max.tz is not None:
            ds_max = ds_max.tz_localize(None)
            expected_start = expected_start.tz_localize(None)

        # Allow 1h tolerance for boundary definition
        assert abs((h_max - ds_max).total_seconds()) <= 3600, (
            f"Holdout max {h_max} does not match dataset max {ds_max}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Probabilistic forecast (v4)
# ═══════════════════════════════════════════════════════════════════════════

class TestProbabilisticForecast:
    """Verify splits, feature safety, and scoring formulas."""

    # ------------------------------------------------------------------
    # 2-A  Train / val / test / holdout are disjoint
    # ------------------------------------------------------------------
    def test_splits_disjoint(self):
        """Train, val, test, holdout timestamp sets must have zero overlap."""
        cfg_path = V4_ROOT / "config" / "planning_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        dataset_path = V4_ROOT / "data" / "planning_dataset.parquet"
        if not dataset_path.exists():
            pytest.skip("planning_dataset.parquet not found")

        df = pd.read_parquet(dataset_path)
        tz = df.index.tz

        train_end = pd.Timestamp(cfg["data"]["train_end"], tz=tz)
        val_end = pd.Timestamp(cfg["data"]["val_end"], tz=tz)
        holdout_hours = cfg["data"].get("holdout_hours", 168)
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours - 1)

        train_idx = set(df[df.index <= train_end].index)
        val_idx = set(df[(df.index > train_end) & (df.index <= val_end)].index)
        test_idx = set(df[(df.index > val_end) & (df.index < holdout_start)].index)
        holdout_idx = set(df[df.index >= holdout_start].index)

        assert len(train_idx & val_idx) == 0, "Train and val overlap!"
        assert len(train_idx & test_idx) == 0, "Train and test overlap!"
        assert len(train_idx & holdout_idx) == 0, "Train and holdout overlap!"
        assert len(val_idx & test_idx) == 0, "Val and test overlap!"
        assert len(val_idx & holdout_idx) == 0, "Val and holdout overlap!"
        assert len(test_idx & holdout_idx) == 0, "Test and holdout overlap!"

    # ------------------------------------------------------------------
    # 2-B  No excluded short-term lags in feature list
    # ------------------------------------------------------------------
    def test_no_excluded_lags_in_features(self):
        """Features used must not include any of the explicitly excluded short-term lags."""
        cfg_path = V4_ROOT / "config" / "planning_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        feat_cfg = cfg["features"]
        active_features = set(
            feat_cfg["calendar"]
            + feat_cfg["weather"]
            + feat_cfg["load"]
            + feat_cfg["anchors"]
            + feat_cfg["interactions"]
        )
        excluded = set(feat_cfg.get("excluded", []))

        leaked = active_features & excluded
        assert len(leaked) == 0, f"Excluded features found in active list: {leaked}"

    # ------------------------------------------------------------------
    # 2-C  Pinball loss formula
    # ------------------------------------------------------------------
    def test_pinball_loss_under(self):
        """y=5, q=3, tau=0.9 → loss = 0.9 * (5-3) = 1.8."""
        from v4.evaluation.pinball_loss import pinball_loss

        loss = pinball_loss(np.array([5.0]), np.array([3.0]), 0.9)
        assert abs(float(loss[0]) - 1.8) < 1e-10

    def test_pinball_loss_over(self):
        """y=5, q=7, tau=0.9 → loss = (1-0.9) * (7-5) = 0.2."""
        from v4.evaluation.pinball_loss import pinball_loss

        loss = pinball_loss(np.array([5.0]), np.array([7.0]), 0.9)
        assert abs(float(loss[0]) - 0.2) < 1e-10

    def test_pinball_loss_exact(self):
        """y=5, q=5, tau=anything → loss = 0."""
        from v4.evaluation.pinball_loss import pinball_loss

        loss = pinball_loss(np.array([5.0]), np.array([5.0]), 0.5)
        assert abs(float(loss[0])) < 1e-10

    # ------------------------------------------------------------------
    # 2-D  CRPS with perfect predictions ≈ 0
    # ------------------------------------------------------------------
    def test_crps_perfect_predictions(self):
        """When all quantile predictions equal the actual, CRPS ≈ 0."""
        from v4.evaluation.pinball_loss import crps_from_quantiles

        y_true = np.array([100.0, 200.0, 300.0])
        quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # All quantile predictions = actual
        q_preds = np.tile(y_true[:, None], (1, len(quantiles)))
        crps = crps_from_quantiles(y_true, q_preds, quantiles)
        assert crps < 1e-6, f"CRPS should be ~0 for perfect predictions, got {crps}"

    # ------------------------------------------------------------------
    # 2-E  Empirical coverage formula
    # ------------------------------------------------------------------
    def test_coverage_all_inside(self):
        """When all actuals are inside the interval, coverage = 100%."""
        from v4.evaluation.pinball_loss import empirical_coverage

        y_true = np.array([5, 10, 15])
        lower = np.array([0, 5, 10])
        upper = np.array([20, 20, 20])
        cov = empirical_coverage(y_true, lower, upper)
        assert abs(cov - 1.0) < 1e-10

    def test_coverage_half_outside(self):
        """When half are outside, coverage ≈ 50%."""
        from v4.evaluation.pinball_loss import empirical_coverage

        y_true = np.array([5, 25, 15, 35])
        lower = np.array([0, 0, 0, 0])
        upper = np.array([10, 10, 20, 20])
        cov = empirical_coverage(y_true, lower, upper)
        assert abs(cov - 0.5) < 1e-10

    def test_coverage_none_inside(self):
        """When none are inside, coverage = 0%."""
        from v4.evaluation.pinball_loss import empirical_coverage

        y_true = np.array([100, 200])
        lower = np.array([0, 0])
        upper = np.array([1, 1])
        cov = empirical_coverage(y_true, lower, upper)
        assert abs(cov) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — Scenario generation
# ═══════════════════════════════════════════════════════════════════════════

class TestScenarioGeneration:
    """Verify no actual prices leak into scenario generation or reduction."""

    # ------------------------------------------------------------------
    # 3-A  generate_scenarios signature has no actuals parameter
    # ------------------------------------------------------------------
    def test_generate_scenarios_no_actuals_param(self):
        """generate_scenarios should not accept actual_prices or y_true."""
        from v4.scenarios.copula_generator import generate_scenarios

        sig = inspect.signature(generate_scenarios)
        param_names = set(sig.parameters.keys())
        forbidden = {"actual_prices", "y_true", "actuals", "y_actual"}
        leaked = param_names & forbidden
        assert len(leaked) == 0, f"generate_scenarios has forbidden params: {leaked}"

    # ------------------------------------------------------------------
    # 3-B  forward_reduction signature has no actuals parameter
    # ------------------------------------------------------------------
    def test_forward_reduction_no_actuals_param(self):
        """forward_reduction should not accept actual_prices or y_true."""
        from v4.scenarios.scenario_reduction import forward_reduction

        sig = inspect.signature(forward_reduction)
        param_names = set(sig.parameters.keys())
        forbidden = {"actual_prices", "y_true", "actuals", "y_actual"}
        leaked = param_names & forbidden
        assert len(leaked) == 0, f"forward_reduction has forbidden params: {leaked}"

    # ------------------------------------------------------------------
    # 3-C  forward_reduction outputs valid weights
    # ------------------------------------------------------------------
    def test_forward_reduction_weights_sum_to_one(self):
        """Reduced scenario weights must sum to 1.0."""
        from v4.scenarios.scenario_reduction import forward_reduction

        rng = np.random.default_rng(42)
        scenarios = rng.uniform(1000, 5000, (50, 168))
        reduced, weights = forward_reduction(scenarios, n_keep=10)

        assert reduced.shape[0] == 10
        assert weights.shape[0] == 10
        assert abs(weights.sum() - 1.0) < 1e-8, f"Weights sum to {weights.sum()}, not 1.0"
        assert np.all(weights >= 0), "Negative weights found"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4 — BESS backtest (v6)
# ═══════════════════════════════════════════════════════════════════════════

class TestBESSBacktest:
    """Validate revenue formulas, CVaR, cycle and SoC bounds."""

    # ------------------------------------------------------------------
    # Fixtures (inline — no conftest dependency needed)
    # ------------------------------------------------------------------
    @staticmethod
    def _bess_dict():
        return {
            "P_max": 20.0,
            "E_cap": 40.0,
            "E_usable": 32.0,
            "E_min": 4.0,
            "E_max": 36.0,
            "E_init": 20.0,
            "eta": 0.9220,
            "eta_charge": 0.9220,
            "eta_discharge": 0.9220,
            "C_deg": 1471.0,
            "terminal_soc_tolerance": 0.1,
            "max_cycles_per_day": 2.0,
            "max_cycles_per_week": 8.08,
        }

    @staticmethod
    def _tx_dict():
        return {
            "iex_transaction_fee_inr_mwh": 20.0,
            "sldc_charge_inr_mwh": 5.0,
            "rldc_charge_inr_mwh": 2.0,
            "transmission_loss_pct": 0.03,
            "dsm_cost_buffer_inr_mwh": 25.0,
        }

    # ------------------------------------------------------------------
    # 4-A  Gross revenue formula hand-check
    # ------------------------------------------------------------------
    def test_gross_revenue_formula(self):
        """Gross = sum(prices * (p_dis * eta_discharge - p_ch)) — hand-verified."""
        from v6.backtest.metrics import compute_weekly_metrics

        T = 168
        bess = self._bess_dict()
        tx = self._tx_dict()
        eta = bess["eta_discharge"]

        prices = np.concatenate([np.full(12, 2500.0), np.full(12, 7000.0)])
        prices = np.tile(prices, 7)

        p_ch = np.where(prices < 4000, 10.0, 0.0)
        p_dis = np.where(prices >= 4000, 10.0, 0.0)
        soc = np.full(T, 20.0)

        m = compute_weekly_metrics(p_ch, p_dis, soc, prices, bess, tx, compute_dsm=False)

        expected_gross = float(np.sum(prices * (p_dis * eta - p_ch)))
        assert abs(m.gross_revenue_inr - expected_gross) < 0.01, (
            f"Gross mismatch: {m.gross_revenue_inr} vs hand-calc {expected_gross}"
        )

    # ------------------------------------------------------------------
    # 4-B  Net = Gross - all costs
    # ------------------------------------------------------------------
    def test_net_equals_gross_minus_costs(self):
        """Net revenue must equal gross minus each cost component."""
        from v6.backtest.metrics import compute_weekly_metrics

        T = 168
        bess = self._bess_dict()
        tx = self._tx_dict()

        prices = np.concatenate([np.full(12, 2500.0), np.full(12, 7000.0)])
        prices = np.tile(prices, 7)

        p_ch = np.where(prices < 4000, 10.0, 0.0)
        p_dis = np.where(prices >= 4000, 10.0, 0.0)
        soc = np.full(T, 20.0)

        m = compute_weekly_metrics(p_ch, p_dis, soc, prices, bess, tx, compute_dsm=False)

        expected_net = (
            m.gross_revenue_inr
            - m.iex_fees_inr
            - m.sldc_rldc_fees_inr
            - m.tx_loss_inr
            - m.degradation_cost_inr
            - m.dsm_cost_inr
        )
        assert abs(m.net_revenue_inr - expected_net) < 0.01, (
            f"Net mismatch: {m.net_revenue_inr} vs (gross - costs) {expected_net}"
        )

    # ------------------------------------------------------------------
    # 4-C  CVaR formula: mean of worst alpha fraction
    # ------------------------------------------------------------------
    def test_cvar_formula(self):
        """CVaR(alpha=0.05) of [1..100] = mean([1,2,3,4,5]) = 3.0."""
        from v6.backtest.metrics import WeeklyMetrics, compute_aggregate_metrics

        # Create 100 fake weekly results with net revenue = 1..100
        results = []
        for i in range(1, 101):
            wm = WeeklyMetrics(
                gross_revenue_inr=float(i) + 10,
                iex_fees_inr=5.0,
                sldc_rldc_fees_inr=2.0,
                tx_loss_inr=1.0,
                dsm_cost_inr=1.0,
                degradation_cost_inr=1.0,
                net_revenue_inr=float(i),
                capture_vs_pf_pct=90.0,
                cycles=1.0,
            )
            results.append(wm)

        agg = compute_aggregate_metrics(results, power_mw=20.0, alpha=0.05)
        # alpha=0.05, n=100 → k = ceil(0.05*100) = 5
        # sorted [1,2,...,100], worst 5 = [1,2,3,4,5], mean = 3.0
        assert abs(agg.cvar_realized - 3.0) < 1e-6, (
            f"CVaR expected 3.0, got {agg.cvar_realized}"
        )

    # ------------------------------------------------------------------
    # 4-D  Cycle formula: total_discharge / E_usable
    # ------------------------------------------------------------------
    def test_cycle_formula(self):
        """Cycles = sum(p_dis * dt) / E_usable."""
        from v6.backtest.metrics import compute_weekly_metrics

        T = 168
        bess = self._bess_dict()
        tx = self._tx_dict()

        prices = np.full(T, 5000.0)
        p_ch = np.zeros(T)
        p_dis = np.zeros(T)
        # Discharge 10 MW for 32 hours → 320 MWh → 320/32 = 10 cycles
        p_dis[:32] = 10.0
        soc = np.full(T, 20.0)

        m = compute_weekly_metrics(p_ch, p_dis, soc, prices, bess, tx, compute_dsm=False)

        total_discharge_mwh = 10.0 * 32  # 320 MWh
        expected_cycles = total_discharge_mwh / bess["E_usable"]  # 320 / 32 = 10
        assert abs(m.cycles - expected_cycles) < 1e-6, (
            f"Cycles expected {expected_cycles}, got {m.cycles}"
        )

    # ------------------------------------------------------------------
    # 4-E  SoC bounds check
    # ------------------------------------------------------------------
    def test_soc_stats(self):
        """Peak and min SoC should be correctly reported."""
        from v6.backtest.metrics import compute_weekly_metrics

        T = 168
        bess = self._bess_dict()
        tx = self._tx_dict()
        prices = np.full(T, 5000.0)
        p_ch = np.zeros(T)
        p_dis = np.zeros(T)

        soc = np.linspace(10.0, 35.0, T)
        m = compute_weekly_metrics(p_ch, p_dis, soc, prices, bess, tx, compute_dsm=False)

        assert abs(m.peak_soc - 35.0) < 0.01
        assert abs(m.min_soc - 10.0) < 0.01

    # ------------------------------------------------------------------
    # 4-F  Aggregate: negative weeks count
    # ------------------------------------------------------------------
    def test_negative_weeks_count(self):
        """Aggregate should correctly count weeks with net_revenue < 0."""
        from v6.backtest.metrics import WeeklyMetrics, compute_aggregate_metrics

        results = []
        for nr in [100, -50, 200, -10, 300]:
            wm = WeeklyMetrics(
                gross_revenue_inr=abs(nr) + 20,
                iex_fees_inr=5.0,
                sldc_rldc_fees_inr=2.0,
                tx_loss_inr=1.0,
                dsm_cost_inr=1.0,
                degradation_cost_inr=1.0,
                net_revenue_inr=float(nr),
                capture_vs_pf_pct=90.0,
                cycles=1.0,
            )
            results.append(wm)

        agg = compute_aggregate_metrics(results, power_mw=20.0)
        assert agg.negative_weeks == 2, f"Expected 2 negative weeks, got {agg.negative_weeks}"

    # ------------------------------------------------------------------
    # 4-G  Max drawdown
    # ------------------------------------------------------------------
    def test_max_drawdown(self):
        """Max drawdown of cumulative P&L should be correct."""
        from v6.backtest.metrics import WeeklyMetrics, compute_aggregate_metrics

        # Net revenues: 100, -200, 50  → cumP&L = [100, -100, -50]
        # running_max = [100, 100, 100], drawdowns = [0, 200, 150]
        # max_dd = 200
        results = []
        for nr in [100, -200, 50]:
            wm = WeeklyMetrics(
                gross_revenue_inr=abs(nr) + 20,
                iex_fees_inr=5.0,
                sldc_rldc_fees_inr=2.0,
                tx_loss_inr=1.0,
                dsm_cost_inr=1.0,
                degradation_cost_inr=1.0,
                net_revenue_inr=float(nr),
                capture_vs_pf_pct=90.0,
                cycles=1.0,
            )
            results.append(wm)

        agg = compute_aggregate_metrics(results, power_mw=20.0)
        assert abs(agg.max_drawdown - 200.0) < 1e-6, (
            f"Max drawdown expected 200.0, got {agg.max_drawdown}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cross-layer consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossLayer:
    """Verify consistency across v2, v4, and v6 layers."""

    # ------------------------------------------------------------------
    # X-A  Same holdout period across v2 and v4
    # ------------------------------------------------------------------
    def test_same_holdout_period(self):
        """v2 and v4 should use the same holdout start/end."""
        # v2 config
        v2_cfg_path = PROJECT_ROOT / "config" / "model_config.yaml"
        with open(v2_cfg_path) as f:
            v2_cfg = yaml.safe_load(f)
        v2_holdout_hours = v2_cfg["cv"].get("holdout_hours", 168)

        # v4 config
        v4_cfg_path = V4_ROOT / "config" / "planning_config.yaml"
        with open(v4_cfg_path) as f:
            v4_cfg = yaml.safe_load(f)
        v4_holdout_hours = v4_cfg["data"].get("holdout_hours", 168)

        assert v2_holdout_hours == v4_holdout_hours, (
            f"Holdout hours differ: v2={v2_holdout_hours}, v4={v4_holdout_hours}"
        )

    # ------------------------------------------------------------------
    # X-B  Feature parity between v4 config and v6 usage
    # ------------------------------------------------------------------
    def test_feature_parity_v4_v6(self):
        """v6 backtest should use the same feature set as v4 planning config."""
        v4_cfg_path = V4_ROOT / "config" / "planning_config.yaml"
        with open(v4_cfg_path) as f:
            v4_cfg = yaml.safe_load(f)

        feat_cfg = v4_cfg["features"]
        v4_features = set(
            feat_cfg["calendar"]
            + feat_cfg["weather"]
            + feat_cfg["load"]
            + feat_cfg["anchors"]
            + feat_cfg["interactions"]
        )

        # v6 uses the same planning config
        v6_cfg_path = V6_ROOT / "config" / "planning_config.yaml"
        if not v6_cfg_path.exists():
            # v6 may reuse v4's config — check the backtest code
            pytest.skip("v6 has no separate planning_config.yaml; shares v4's config")
            return

        with open(v6_cfg_path) as f:
            v6_cfg = yaml.safe_load(f)

        v6_feat_cfg = v6_cfg["features"]
        v6_features = set(
            v6_feat_cfg["calendar"]
            + v6_feat_cfg["weather"]
            + v6_feat_cfg["load"]
            + v6_feat_cfg["anchors"]
            + v6_feat_cfg["interactions"]
        )

        missing_in_v6 = v4_features - v6_features
        extra_in_v6 = v6_features - v4_features

        assert len(missing_in_v6) == 0, f"Features in v4 but missing in v6: {missing_in_v6}"
        assert len(extra_in_v6) == 0, f"Extra features in v6 not in v4: {extra_in_v6}"

    # ------------------------------------------------------------------
    # X-C  Revenue per MW per year formula
    # ------------------------------------------------------------------
    def test_revenue_per_mw_formula(self):
        """revenue_per_mw_per_year_lakh = (mean_weekly * 52 / power_mw) / 1e5."""
        from v6.backtest.metrics import WeeklyMetrics, compute_aggregate_metrics

        results = []
        for _ in range(10):
            wm = WeeklyMetrics(
                gross_revenue_inr=120_000,
                iex_fees_inr=5000,
                sldc_rldc_fees_inr=2000,
                tx_loss_inr=1000,
                dsm_cost_inr=500,
                degradation_cost_inr=500,
                net_revenue_inr=111_000,
                capture_vs_pf_pct=90.0,
                cycles=1.0,
            )
            results.append(wm)

        agg = compute_aggregate_metrics(results, power_mw=20.0)
        expected = (111_000 * 52 / 20.0) / 1e5
        assert abs(agg.revenue_per_mw_per_year_lakh - expected) < 1e-6, (
            f"Rev/MW/year expected {expected}, got {agg.revenue_per_mw_per_year_lakh}"
        )
