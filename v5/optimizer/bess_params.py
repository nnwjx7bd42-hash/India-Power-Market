"""
Load and validate BESS battery parameters from the optimizer config YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_CFG = Path(__file__).resolve().parent.parent / "config" / "optimizer_config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read the full optimizer config YAML and return as dict."""
    path = Path(path) if path else _DEFAULT_CFG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# BESS parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class BESSParams:
    """Physical and economic battery parameters."""

    power_mw: float = 10.0
    energy_mwh: float = 20.0
    efficiency_oneway: float = 0.92
    degradation_cost_inr_per_mwh: float = 1471.0
    initial_soc_fraction: float = 0.5
    min_soc_fraction: float = 0.1
    max_soc_fraction: float = 0.95
    terminal_soc_tolerance: float = 0.1

    # --- derived (computed in __post_init__) ---
    eta: float = field(init=False)
    P_max: float = field(init=False)
    E_cap: float = field(init=False)
    E_min: float = field(init=False)
    E_max: float = field(init=False)
    E_init: float = field(init=False)
    C_deg: float = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        self.eta = self.efficiency_oneway
        self.P_max = self.power_mw
        self.E_cap = self.energy_mwh
        self.E_min = self.min_soc_fraction * self.energy_mwh
        self.E_max = self.max_soc_fraction * self.energy_mwh
        self.E_init = self.initial_soc_fraction * self.energy_mwh
        self.C_deg = self.degradation_cost_inr_per_mwh

    def _validate(self) -> None:
        """Raise if parameters are physically invalid."""
        if self.power_mw <= 0:
            raise ValueError(f"power_mw must be >0, got {self.power_mw}")
        if self.energy_mwh <= 0:
            raise ValueError(f"energy_mwh must be >0, got {self.energy_mwh}")
        if not (0 < self.efficiency_oneway <= 1.0):
            raise ValueError(f"efficiency_oneway must be in (0,1], got {self.efficiency_oneway}")
        if not (0 <= self.min_soc_fraction < self.max_soc_fraction <= 1.0):
            raise ValueError(
                f"SoC fractions invalid: min={self.min_soc_fraction}, max={self.max_soc_fraction}"
            )
        if not (self.min_soc_fraction <= self.initial_soc_fraction <= self.max_soc_fraction):
            raise ValueError(
                f"initial_soc_fraction {self.initial_soc_fraction} outside "
                f"[{self.min_soc_fraction}, {self.max_soc_fraction}]"
            )
        if not (0 <= self.terminal_soc_tolerance <= 1.0):
            raise ValueError(f"terminal_soc_tolerance must be in [0,1], got {self.terminal_soc_tolerance}")

    @classmethod
    def from_config(cls, cfg: Dict[str, Any] | None = None, config_path: str | Path | None = None) -> "BESSParams":
        """Build from the 'bess' section of the optimizer config."""
        if cfg is None:
            cfg = load_config(config_path)
        bess_cfg = cfg.get("bess", {})
        return cls(
            power_mw=float(bess_cfg.get("power_mw", 10)),
            energy_mwh=float(bess_cfg.get("energy_mwh", 20)),
            efficiency_oneway=float(bess_cfg.get("efficiency_oneway", 0.92)),
            degradation_cost_inr_per_mwh=float(bess_cfg.get("degradation_cost_inr_per_mwh", 1471)),
            initial_soc_fraction=float(bess_cfg.get("initial_soc_fraction", 0.5)),
            min_soc_fraction=float(bess_cfg.get("min_soc_fraction", 0.1)),
            max_soc_fraction=float(bess_cfg.get("max_soc_fraction", 0.95)),
            terminal_soc_tolerance=float(bess_cfg.get("terminal_soc_tolerance", 0.1)),
        )

    def as_dict(self) -> Dict[str, float]:
        """Return all derived parameters as a flat dict (for solver functions)."""
        return {
            "P_max": self.P_max,
            "E_cap": self.E_cap,
            "E_min": self.E_min,
            "E_max": self.E_max,
            "E_init": self.E_init,
            "eta": self.eta,
            "C_deg": self.C_deg,
            "terminal_soc_tolerance": self.terminal_soc_tolerance,
        }
