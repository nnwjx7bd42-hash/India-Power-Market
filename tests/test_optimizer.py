"""Smoke tests for the v6 BESS optimisers."""
from __future__ import annotations

import numpy as np
import pytest


# ── Deterministic LP ─────────────────────────────────────────────────────

class TestDeterministicLP:
    """Basic feasibility and constraint tests for the deterministic LP."""

    def test_feasible_24h(self, bess_dict, transaction_costs_dict, synthetic_prices_24h):
        from v6.optimizer.deterministic_lp import solve_deterministic_lp

        result = solve_deterministic_lp(
            synthetic_prices_24h, bess_dict, transaction_costs_dict, "highs",
        )
        assert result.status in ("optimal", "feasible")
        assert len(result.p_ch) == 24
        assert len(result.p_dis) == 24
        assert len(result.soc) == 24

    def test_soc_within_bounds(self, bess_dict, transaction_costs_dict, synthetic_prices_24h):
        from v6.optimizer.deterministic_lp import solve_deterministic_lp

        result = solve_deterministic_lp(
            synthetic_prices_24h, bess_dict, transaction_costs_dict, "highs",
        )
        assert np.all(result.soc >= bess_dict["E_min"] - 0.01)
        assert np.all(result.soc <= bess_dict["E_max"] + 0.01)

    def test_positive_revenue_with_spread(self, bess_dict, transaction_costs_dict, synthetic_prices_24h):
        """With a clear 2500→7000 spread, optimizer should earn positive revenue."""
        from v6.optimizer.deterministic_lp import solve_deterministic_lp

        result = solve_deterministic_lp(
            synthetic_prices_24h, bess_dict, transaction_costs_dict, "highs",
        )
        assert result.revenue > 0, "Should profit from obvious arbitrage spread"

    def test_cycle_constraint_enforced(self, bess_dict, transaction_costs_dict, synthetic_prices_168h):
        """Weekly total discharge energy should respect cycle limit."""
        from v6.optimizer.deterministic_lp import solve_deterministic_lp

        result = solve_deterministic_lp(
            synthetic_prices_168h, bess_dict, transaction_costs_dict, "highs",
        )
        total_discharge_mwh = float(np.sum(result.p_dis))
        max_allowed = bess_dict["max_cycles_per_week"] * bess_dict["E_usable"]
        assert total_discharge_mwh <= max_allowed + 0.01, (
            f"Weekly discharge {total_discharge_mwh:.1f} MWh exceeds "
            f"cycle limit {max_allowed:.1f} MWh"
        )


# ── Stochastic CVaR LP ──────────────────────────────────────────────────

class TestStochasticCVaR:
    """Basic shape and feasibility tests for the stochastic CVaR LP."""

    @staticmethod
    def _make_scenarios(n_scenarios: int = 5, T: int = 24) -> tuple:
        rng = np.random.default_rng(42)
        daily = np.concatenate([np.full(12, 2500.0), np.full(12, 7000.0)])
        n_days = max(1, T // 24)
        base = np.tile(daily, n_days)[:T]
        scenarios = np.array([base + rng.normal(0, 200, T) for _ in range(n_scenarios)])
        scenarios = np.clip(scenarios, 0, None)
        weights = np.ones(n_scenarios) / n_scenarios
        return scenarios, weights

    def test_feasible_5_scenarios(self, bess_dict, transaction_costs_dict):
        from v6.optimizer.stochastic_cvar import solve_stochastic_cvar

        scenarios, weights = self._make_scenarios(5, 24)
        result = solve_stochastic_cvar(
            scenarios, weights, bess_dict, transaction_costs_dict,
            beta=0.3, alpha=0.95, solver_name="highs",
        )
        assert result.status in ("optimal", "feasible")

    def test_output_shapes(self, bess_dict, transaction_costs_dict):
        from v6.optimizer.stochastic_cvar import solve_stochastic_cvar

        S, T = 5, 24
        scenarios, weights = self._make_scenarios(S, T)
        result = solve_stochastic_cvar(
            scenarios, weights, bess_dict, transaction_costs_dict,
        )
        assert result.p_ch.shape == (T,)
        assert result.p_dis.shape == (T,)
        assert result.soc.shape == (T,)
        assert result.per_scenario_revenue.shape == (S,)

    def test_cycle_constraint_enforced(self, bess_dict, transaction_costs_dict):
        """Weekly cycle limit is a hard constraint across all scenarios."""
        from v6.optimizer.stochastic_cvar import solve_stochastic_cvar

        scenarios, weights = self._make_scenarios(5, 168)
        result = solve_stochastic_cvar(
            scenarios, weights, bess_dict, transaction_costs_dict,
        )
        total_discharge = float(np.sum(result.p_dis))
        max_allowed = bess_dict["max_cycles_per_week"] * bess_dict["E_usable"]
        assert total_discharge <= max_allowed + 0.01, (
            f"Stochastic LP weekly discharge {total_discharge:.1f} MWh > "
            f"cycle limit {max_allowed:.1f} MWh"
        )
