"""
Convert hourly MW schedule to IEX DAM 15-minute bid format.

IEX DAM operates in 96 × 15-minute blocks per day.
Each block requires buy_mw and sell_mw bids, rounded to 0.1 MW steps.
"""
from __future__ import annotations

from typing import List, Dict


def hourly_to_15min_bids(
    P_charge_hourly: list | tuple,
    P_discharge_hourly: list | tuple,
    P_max: float,
    volume_step_mw: float = 0.1,
) -> List[Dict[str, float]]:
    """
    Convert hourly MW schedule to 96 × 15-minute bid volumes.
    
    Parameters
    ----------
    P_charge_hourly : (24,) hourly charge power MW
    P_discharge_hourly : (24,) hourly discharge power MW
    P_max : maximum power rating MW
    volume_step_mw : bid volume granularity (default 0.1 MW)
    
    Returns
    -------
    List of 96 dicts with keys: 'block', 'buy_mw', 'sell_mw'
    Blocks are numbered 0-95 (00:00-00:15 = block 0, ..., 23:45-24:00 = block 95)
    """
    bids = []
    
    for h in range(24):
        # Each hour has 4 × 15-minute blocks
        mw_charge = P_charge_hourly[h] if h < len(P_charge_hourly) else 0.0
        mw_discharge = P_discharge_hourly[h] if h < len(P_discharge_hourly) else 0.0
        
        # Distribute evenly across 4 blocks (can be improved with intra-hour profile)
        charge_per_block = mw_charge / 4.0
        discharge_per_block = mw_discharge / 4.0
        
        for q in range(4):  # 4 quarters per hour
            block_num = h * 4 + q
            
            # Round to nearest volume_step
            buy_mw = round(charge_per_block / volume_step_mw) * volume_step_mw
            sell_mw = round(discharge_per_block / volume_step_mw) * volume_step_mw
            
            # Clamp to P_max
            buy_mw = min(buy_mw, P_max)
            sell_mw = min(sell_mw, P_max)
            
            # Ensure non-negative
            buy_mw = max(0.0, buy_mw)
            sell_mw = max(0.0, sell_mw)
            
            bids.append({
                "block": block_num,
                "buy_mw": buy_mw,
                "sell_mw": sell_mw,
            })
    
    return bids


def bids_to_hourly_schedule(bids: List[Dict[str, float]]) -> tuple:
    """
    Convert 96-block bids back to hourly schedule (for validation).
    
    Returns
    -------
    (P_charge_hourly, P_discharge_hourly) as (24,) arrays
    """
    P_charge = [0.0] * 24
    P_discharge = [0.0] * 24
    
    for bid in bids:
        block = bid["block"]
        hour = block // 4
        if 0 <= hour < 24:
            P_charge[hour] += bid["buy_mw"]
            P_discharge[hour] += bid["sell_mw"]
    
    return tuple(P_charge), tuple(P_discharge)
