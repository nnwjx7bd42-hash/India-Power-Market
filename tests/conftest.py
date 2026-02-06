"""Shared fixtures for the test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Make project packages importable ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "v4"), str(_PROJECT_ROOT / "v5"), str(_PROJECT_ROOT / "v6")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Synthetic BESS params dict ───────────────────────────────────────────
@pytest.fixture
def bess_dict() -> dict:
    """Minimal BESS parameter dict matching v6 optimizer interface."""
    return {
        "P_max": 20.0,           # MW
        "E_cap": 40.0,           # MWh
        "E_usable": 32.0,        # MWh  (80% of E_cap)
        "E_min": 4.0,            # MWh  (10% SoC)
        "E_max": 36.0,           # MWh  (90% SoC)
        "E_init": 20.0,          # MWh  (50% SoC)
        "eta": 0.9220,
        "eta_charge": 0.9220,
        "eta_discharge": 0.9220,
        "C_deg": 1471.0,         # INR/MWh throughput degradation cost
        "terminal_soc_tolerance": 0.1,
        "max_cycles_per_day": 2.0,
        "max_cycles_per_week": 8.08,
    }


@pytest.fixture
def transaction_costs_dict() -> dict:
    """Minimal transaction cost dict."""
    return {
        "iex_transaction_fee_inr_mwh": 20.0,
        "sldc_charge_inr_mwh": 5.0,
        "rldc_charge_inr_mwh": 2.0,
        "transmission_loss_pct": 0.03,
        "dsm_cost_buffer_inr_mwh": 25.0,
    }


@pytest.fixture
def synthetic_prices_24h() -> np.ndarray:
    """24-hour synthetic price curve with clear peak/off-peak pattern."""
    # Off-peak: ~2500, peak: ~7000
    off_peak = np.full(12, 2500.0)  # hours 0-11
    peak = np.full(12, 7000.0)      # hours 12-23
    return np.concatenate([off_peak, peak])


@pytest.fixture
def synthetic_prices_168h() -> np.ndarray:
    """168-hour (1 week) synthetic prices repeating daily pattern."""
    daily = np.concatenate([np.full(12, 2500.0), np.full(12, 7000.0)])
    return np.tile(daily, 7)
