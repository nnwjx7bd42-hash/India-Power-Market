"""Basic constraint verification tests for BESS optimizer."""
import pytest
from src.optimizer.two_stage_bess import TwoStageBESS
from src.optimizer.bess_params import BESSParams


def test_soc_bounds_respected():
    """Verify SoC never violates physical limits (20-180 MWh)."""
    params = BESSParams(
        p_max_mw=50,
        e_max_mwh=180,
        e_min_mwh=20,
        eta_charge=0.9,
        eta_discharge=0.9,
        soc_initial_mwh=100.0,
        soc_terminal_min_mwh=100.0,
        degradation_cost_rs_mwh=0.0,
        iex_fee_rs_mwh=0.0
    )
    # Create fake 24-hour price scenarios (extreme spread to stress bounds)
    # Shape: (1, 24) for 1 scenario
    import numpy as np
    dam_prices = np.array([[2000.0]*12 + [8000.0]*12])  # Rs/MWh
    rtm_prices = np.array([[2500.0]*12 + [7500.0]*12])
    
    config = {'solver': 'PULP_CBC_CMD', 'lambda_risk': 0.0, 'lambda_dev': 0.0}
    optimizer = TwoStageBESS(params, config)
    
    result = optimizer.solve(dam_prices, rtm_prices)
    
    assert result['status'] == 'Optimal', "Optimizer failed to solve"
    
    # THE ACTUAL CONSTRAINT CHECK
    soc_trajectory = result['scenarios'][0]['soc']
    # soc has 25 points (0..24)
    for hour in range(25):
        assert soc_trajectory[hour] >= params.e_min_mwh - 1e-5, f"SoC below min at hour {hour}: {soc_trajectory[hour]}"
        assert soc_trajectory[hour] <= params.e_max_mwh + 1e-5, f"SoC above max at hour {hour}: {soc_trajectory[hour]}"


def test_ramp_rate_constraints():
    """Verify ramp rates don't exceed ±50 MW/hour."""
    # Placeholder — expand with actual optimizer instantiation
    pass
