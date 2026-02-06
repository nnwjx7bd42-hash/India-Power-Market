"""
Load and validate BESS battery parameters from the optimizer config YAML.
V6: Extended with cycle limits, separate charge/discharge efficiencies, degradation tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_BESS_CFG = Path(__file__).resolve().parent.parent / "config" / "bess_config.yaml"
_DEFAULT_OPT_CFG = Path(__file__).resolve().parent.parent / "config" / "optimizer_config.yaml"


def load_bess_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read the BESS config YAML and return as dict."""
    path = Path(path) if path else _DEFAULT_BESS_CFG
    if not path.exists():
        raise FileNotFoundError(f"BESS config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read the optimizer config YAML and return as dict."""
    path = Path(path) if path else _DEFAULT_OPT_CFG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# BESS parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class BESSParams:
    """Physical and economic battery parameters (V6: extended)."""

    # Power and energy
    P_max_mw: float = 20.0
    E_cap_mwh: float = 40.0
    E_usable_mwh: float = 32.0  # computed from soc_min/max if not provided
    
    # Efficiencies (separate charge/discharge)
    eta_charge: float = 0.9220
    eta_discharge: float = 0.9220
    rte_year1: float = 0.85
    
    # Degradation
    rte_degradation_per_year: float = 0.0025
    capacity_fade_per_year: float = 0.02
    degradation_cost_inr_per_mwh: float = 1471.0
    
    # SOC bounds
    soc_min_pct: float = 10.0
    soc_max_pct: float = 90.0
    E_init_mwh: float = 20.0
    terminal_soc_tolerance: float = 0.1
    
    # Cycle limits
    max_cycles_per_day: float = 2.0
    max_cycles_per_week: float = 8.08
    max_cycles_per_year: float = 420.0
    
    # Availability
    min_annual_availability: float = 0.95
    planned_outage_hours_per_year: float = 168.0

    # --- derived (computed in __post_init__) ---
    eta: float = field(init=False)  # backward compat: use eta_charge
    P_max: float = field(init=False)
    E_cap: float = field(init=False)
    E_min: float = field(init=False)
    E_max: float = field(init=False)
    E_init: float = field(init=False)
    C_deg: float = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        # Backward compatibility: eta defaults to charge efficiency
        self.eta = self.eta_charge
        self.P_max = self.P_max_mw
        self.E_cap = self.E_cap_mwh
        
        # Compute E_usable if not explicitly set
        if self.E_usable_mwh <= 0:
            self.E_usable_mwh = self.E_cap_mwh * (self.soc_max_pct - self.soc_min_pct) / 100.0
        
        self.E_min = self.soc_min_pct / 100.0 * self.E_cap_mwh
        self.E_max = self.soc_max_pct / 100.0 * self.E_cap_mwh
        self.E_init = self.E_init_mwh
        self.C_deg = self.degradation_cost_inr_per_mwh

    def _validate(self) -> None:
        """Raise if parameters are physically invalid."""
        if self.P_max_mw <= 0:
            raise ValueError(f"P_max_mw must be >0, got {self.P_max_mw}")
        if self.E_cap_mwh <= 0:
            raise ValueError(f"E_cap_mwh must be >0, got {self.E_cap_mwh}")
        if not (0 < self.eta_charge <= 1.0):
            raise ValueError(f"eta_charge must be in (0,1], got {self.eta_charge}")
        if not (0 < self.eta_discharge <= 1.0):
            raise ValueError(f"eta_discharge must be in (0,1], got {self.eta_discharge}")
        if not (0 <= self.soc_min_pct < self.soc_max_pct <= 100.0):
            raise ValueError(
                f"SoC percentages invalid: min={self.soc_min_pct}, max={self.soc_max_pct}"
            )
        if not (self.soc_min_pct <= (self.E_init_mwh / self.E_cap_mwh * 100) <= self.soc_max_pct):
            raise ValueError(
                f"E_init_mwh {self.E_init_mwh} corresponds to "
                f"{self.E_init_mwh/self.E_cap_mwh*100:.1f}% SOC, outside "
                f"[{self.soc_min_pct}, {self.soc_max_pct}]"
            )
        if not (0 <= self.terminal_soc_tolerance <= 1.0):
            raise ValueError(f"terminal_soc_tolerance must be in [0,1], got {self.terminal_soc_tolerance}")
        if self.max_cycles_per_day <= 0 or self.max_cycles_per_week <= 0:
            raise ValueError("Cycle limits must be >0")

    @classmethod
    def from_config(cls, cfg: Dict[str, Any] | None = None, config_path: str | Path | None = None) -> "BESSParams":
        """Build from the 'bess' section of the BESS config."""
        if cfg is None:
            cfg = load_bess_config(config_path)
        bess_cfg = cfg.get("bess", {})
        cycle_cfg = cfg.get("cycle_limits", {})
        
        return cls(
            P_max_mw=float(bess_cfg.get("P_max_mw", 20.0)),
            E_cap_mwh=float(bess_cfg.get("E_cap_mwh", 40.0)),
            E_usable_mwh=float(bess_cfg.get("E_usable_mwh", 0.0)),  # 0 = compute from soc bounds
            eta_charge=float(bess_cfg.get("eta_charge", 0.9220)),
            eta_discharge=float(bess_cfg.get("eta_discharge", 0.9220)),
            rte_year1=float(bess_cfg.get("rte_year1", 0.85)),
            rte_degradation_per_year=float(bess_cfg.get("rte_degradation_per_year", 0.0025)),
            capacity_fade_per_year=float(bess_cfg.get("capacity_fade_per_year", 0.02)),
            degradation_cost_inr_per_mwh=float(bess_cfg.get("degradation_cost_inr_per_mwh", 1471.0)),
            soc_min_pct=float(bess_cfg.get("soc_min_pct", 10.0)),
            soc_max_pct=float(bess_cfg.get("soc_max_pct", 90.0)),
            E_init_mwh=float(bess_cfg.get("E_init_mwh", 20.0)),
            terminal_soc_tolerance=float(bess_cfg.get("terminal_soc_tolerance", 0.1)),
            max_cycles_per_day=float(cycle_cfg.get("max_cycles_per_day", 2.0)),
            max_cycles_per_week=float(cycle_cfg.get("max_cycles_per_week", 8.08)),
            max_cycles_per_year=float(cycle_cfg.get("max_cycles_per_year", 420.0)),
        )

    def get_bess_params_for_year(self, year_offset: float) -> "BESSParams":
        """
        Return degraded BESS parameters after year_offset years from commissioning.
        
        Parameters
        ----------
        year_offset : years since commissioning (0 = Year 1)
        
        Returns
        -------
        New BESSParams instance with degraded capacity and efficiency
        """
        # RTE degradation: Year 1 RTE drops by degradation_per_year
        rte_current = max(0.7, self.rte_year1 - year_offset * self.rte_degradation_per_year)
        # Split RTE into charge/discharge: sqrt(RTE) for each
        eta_charge_new = rte_current ** 0.5
        eta_discharge_new = rte_current ** 0.5
        
        # Capacity fade: E_cap reduces by fade_per_year
        fade = min(0.3, year_offset * self.capacity_fade_per_year)  # cap at 30% fade
        E_cap_new = self.E_cap_mwh * (1 - fade)
        
        # Create new instance with degraded params
        params = BESSParams(
            P_max_mw=self.P_max_mw,
            E_cap_mwh=E_cap_new,
            E_usable_mwh=0.0,  # will be recomputed in __post_init__
            eta_charge=eta_charge_new,
            eta_discharge=eta_discharge_new,
            rte_year1=rte_current,
            rte_degradation_per_year=self.rte_degradation_per_year,
            capacity_fade_per_year=self.capacity_fade_per_year,
            degradation_cost_inr_per_mwh=self.degradation_cost_inr_per_mwh,
            soc_min_pct=self.soc_min_pct,
            soc_max_pct=self.soc_max_pct,
            E_init_mwh=self.E_init_mwh,
            terminal_soc_tolerance=self.terminal_soc_tolerance,
            max_cycles_per_day=self.max_cycles_per_day,
            max_cycles_per_week=self.max_cycles_per_week,
            max_cycles_per_year=self.max_cycles_per_year,
        )
        return params

    def as_dict(self) -> Dict[str, float]:
        """Return all derived parameters as a flat dict (for solver functions)."""
        return {
            "P_max": self.P_max,
            "E_cap": self.E_cap,
            "E_usable": self.E_usable_mwh,
            "E_min": self.E_min,
            "E_max": self.E_max,
            "E_init": self.E_init,
            "eta": self.eta,  # backward compat
            "eta_charge": self.eta_charge,
            "eta_discharge": self.eta_discharge,
            "C_deg": self.C_deg,
            "terminal_soc_tolerance": self.terminal_soc_tolerance,
            "max_cycles_per_day": self.max_cycles_per_day,
            "max_cycles_per_week": self.max_cycles_per_week,
        }
