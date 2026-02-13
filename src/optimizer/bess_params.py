from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class BESSParams:
    p_max_mw: float
    e_max_mwh: float
    e_min_mwh: float
    eta_charge: float
    eta_discharge: float
    soc_initial_mwh: float
    soc_terminal_min_mwh: float
    degradation_cost_rs_mwh: float
    iex_fee_rs_mwh: float
    max_cycles_per_day: Optional[float] = None

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
