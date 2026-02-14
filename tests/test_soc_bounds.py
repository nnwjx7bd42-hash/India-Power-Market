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

    # VISUAL PROOF (Generated on request)
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        hours = range(24)
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Price (Rs/MWh)', color='tab:red')
        ax1.step(hours, dam_prices.flatten(), where='post', color='tab:red', label='Extreme Pricing', alpha=0.6)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('SoC (MWh)', color='tab:purple')
        ax2.plot(range(25), soc_trajectory, color='tab:purple', linewidth=2, label='SoC Response', marker='o')
        ax2.tick_params(axis='y', labelcolor='tab:purple')
        ax2.set_ylim(0, 200)
        
        # Draw Bounds
        ax2.axhline(params.e_min_mwh, color='k', linestyle='--', label='Min Limit (20)')
        ax2.axhline(params.e_max_mwh, color='k', linestyle='--', label='Max Limit (180)')
        
        plt.title('Optimizer Stress Test: Extreme Pricing vs SoC Bounds')
        fig.tight_layout()
        plt.savefig('tests/stress_test_soc.png')
        plt.close()
    except ImportError:
        pass



def test_ramp_rate_constraints():
    """Verify ramp rates don't exceed ±50 MW/hour."""
    # Placeholder — expand with actual optimizer instantiation
    pass
