import numpy as np
import yaml

class CostModel:
    """Pluggable trading cost model for BESS operations."""

    def __init__(self, config: dict):
        """
        Parse costs_config.yaml and store parameters.
        """
        self.config = config.get('costs', {})
        
        # Extract component configs
        self.iex_cfg = self.config.get('iex_transaction_fee', {})
        self.sched_cfg = self.config.get('scheduling_charges', {})
        self.deg_cfg = self.config.get('degradation', {})
        self.ists_cfg = self.config.get('ists_charges', {})
        self.dsm_cfg = self.config.get('dsm_penalties', {})
        self.oa_cfg = self.config.get('open_access', {})

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(data)

    def compute_costs(self, charge: np.ndarray, discharge: np.ndarray,
                      dam_actual: np.ndarray = None, rtm_actual: np.ndarray = None, 
                      scheduled: np.ndarray = None) -> dict:
        """
        Given a 24-hour dispatch schedule, compute all trading and operational costs.
        
        Parameters:
        - charge: ndarray(24) — MW charged (positive values)
        - discharge: ndarray(24) — MW discharged (positive values)
        - scheduled: ndarray(24) — Scheduled volume (for DSM), defaults to actual if None.
        """
        energy_charged = np.sum(charge)
        energy_discharged = np.sum(discharge)
        total_throughput = energy_charged + energy_discharged
        
        # 1. IEX Transaction Fee (both sides)
        iex_fee = 0
        if self.iex_cfg.get('enabled'):
            iex_fee = total_throughput * self.iex_cfg.get('fee_per_mwh_per_side', 0)
            
        # 2. Scheduling Charges
        scheduling = 0
        if self.sched_cfg.get('enabled'):
            sldc = self.sched_cfg.get('sldc_per_day', 0)
            rldc = total_throughput * self.sched_cfg.get('rldc_per_mwh', 0)
            scheduling = sldc + rldc
            
        # 3. Battery Degradation
        degradation = 0
        if self.deg_cfg.get('enabled'):
            # Model: cost applies to discharge throughput (1 cycle = capacity discharged once)
            degradation = energy_discharged * self.deg_cfg.get('cost_per_mwh_throughput', 0)
            
        # 4. ISTS Charges
        ists = 0
        if self.ists_cfg.get('enabled') and not self.ists_cfg.get('waiver'):
            ists = total_throughput * self.ists_cfg.get('charge_per_mwh', 0)
            
        # 5. DSM Penalty Estimates
        dsm_penalty = 0
        if self.dsm_cfg.get('enabled'):
            if scheduled is not None:
                # Actual dev = abs(actual physical - scheduled)
                # actual physical = discharge - charge
                actual_net = discharge - charge
                dev_mwh = np.sum(np.abs(actual_net - scheduled))
            else:
                # Approximation: assume fixed % of throughput deviates
                dev_mwh = total_throughput * (self.dsm_cfg.get('assumed_deviation_pct', 0) / 100.0)
            
            dsm_penalty = dev_mwh * self.dsm_cfg.get('estimated_penalty_per_mwh_deviation', 0)
            
        # 6. Open Access
        oa = 0
        if self.oa_cfg.get('enabled'):
            oa = energy_charged * (self.oa_cfg.get('cross_subsidy_surcharge_per_mwh', 0) + 
                                  self.oa_cfg.get('additional_surcharge_per_mwh', 0))
            
        total_costs = iex_fee + scheduling + degradation + ists + dsm_penalty + oa
        
        return {
            'iex_transaction_fee': iex_fee,
            'scheduling_charges': scheduling,
            'degradation_cost': degradation,
            'ists_charges': ists,
            'dsm_penalty_estimate': dsm_penalty,
            'open_access_charges': oa,
            'total_costs': total_costs,
            'cost_breakdown_pct': {
                'iex': (iex_fee / total_costs * 100) if total_costs > 0 else 0,
                'scheduling': (scheduling / total_costs * 100) if total_costs > 0 else 0,
                'degradation': (degradation / total_costs * 100) if total_costs > 0 else 0,
                'dsm': (dsm_penalty / total_costs * 100) if total_costs > 0 else 0
            }
        }
