"""
Transaction costs for IEX DAM trading: IEX fees, SLDC/RLDC charges, transmission losses.

All costs that reduce net revenue per MWh traded.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_transaction_costs(
    P_dis: float,
    P_ch: float,
    price: float,
    dt: float,
    costs_dict: Dict[str, float],
    eta_discharge: float = 0.9220,
) -> Tuple[float, float, float]:
    """
    Compute transaction costs for a single timestep.
    
    Parameters
    ----------
    P_dis : discharge power MW
    P_ch  : charge power MW
    price : clearing price INR/MWh
    dt    : timestep duration hours
    costs_dict : dict with keys:
        - 'iex_transaction_fee_inr_mwh': 20.0
        - 'sldc_charge_inr_mwh': 5.0
        - 'rldc_charge_inr_mwh': 2.0
        - 'transmission_loss_pct': 0.03
    eta_discharge : discharge efficiency
    
    Returns
    -------
    (buy_cost, sell_cost, tx_loss_value) tuple in INR
    """
    fee_per_mwh = (
        costs_dict.get("iex_transaction_fee_inr_mwh", 20.0)
        + costs_dict.get("sldc_charge_inr_mwh", 5.0)
        + costs_dict.get("rldc_charge_inr_mwh", 2.0)
    )
    
    # Buy side: energy drawn from grid
    energy_drawn_mwh = P_ch * dt
    buy_cost = energy_drawn_mwh * (price + fee_per_mwh)
    
    # Sell side: effective injection after discharge efficiency and tx losses
    tx_loss_pct = costs_dict.get("transmission_loss_pct", 0.03)
    effective_injection_mwh = P_dis * eta_discharge * (1 - tx_loss_pct) * dt
    sell_revenue_gross = effective_injection_mwh * price
    sell_cost_fees = effective_injection_mwh * fee_per_mwh
    sell_cost = -sell_revenue_gross + sell_cost_fees  # negative = revenue, positive = cost
    
    # Transmission loss value (opportunity cost)
    tx_loss_value = P_dis * eta_discharge * tx_loss_pct * dt * price
    
    return buy_cost, sell_cost, tx_loss_value


def net_revenue_per_timestep(
    P_dis: float,
    P_ch: float,
    price: float,
    dt: float,
    costs_dict: Dict[str, float],
    bess_dict: Dict[str, float],
) -> float:
    """
    Compute net revenue per timestep including all costs.
    
    Formula:
        sell_revenue - buy_cost - degradation - dsm_buffer
    
    Parameters
    ----------
    P_dis : discharge power MW
    P_ch  : charge power MW
    price : clearing price INR/MWh
    dt    : timestep duration hours
    costs_dict : transaction costs dict (from transaction_costs.yaml)
    bess_dict : BESS parameters dict (from bess.as_dict())
    
    Returns
    -------
    Net revenue in INR (positive = profit, negative = loss)
    """
    eta_discharge = bess_dict.get("eta_discharge", bess_dict.get("eta", 0.9220))
    C_deg = bess_dict.get("C_deg", 1471.0)
    dsm_buffer = costs_dict.get("dsm_cost_buffer_inr_mwh", 25.0)
    
    # Transaction costs
    buy_cost, sell_cost, tx_loss_value = compute_transaction_costs(
        P_dis, P_ch, price, dt, costs_dict, eta_discharge
    )
    
    # Sell revenue (after tx losses and fees)
    effective_injection_mwh = P_dis * eta_discharge * (1 - costs_dict.get("transmission_loss_pct", 0.03)) * dt
    sell_revenue = effective_injection_mwh * price
    
    # Buy cost (energy drawn + fees)
    energy_drawn_mwh = P_ch * dt
    buy_cost_total = energy_drawn_mwh * (price + costs_dict.get("iex_transaction_fee_inr_mwh", 20.0) +
                                         costs_dict.get("sldc_charge_inr_mwh", 5.0) +
                                         costs_dict.get("rldc_charge_inr_mwh", 2.0))
    
    # Degradation cost (throughput-based)
    degradation_cost = (P_dis * dt + P_ch * dt) * C_deg / 1000.0  # C_deg in INR/MWh
    
    # DSM buffer cost
    dsm_cost = (P_dis * dt + P_ch * dt) * dsm_buffer / 1000.0
    
    net_revenue = sell_revenue - buy_cost_total - degradation_cost - dsm_cost
    
    return net_revenue


def compute_total_transaction_costs(
    P_dis: np.ndarray,
    P_ch: np.ndarray,
    prices: np.ndarray,
    dt: float,
    costs_dict: Dict[str, float],
    bess_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute total transaction costs over a full schedule.
    
    Returns
    -------
    dict with keys:
        - 'iex_fees_inr': total IEX transaction fees
        - 'sldc_rldc_fees_inr': total SLDC + RLDC charges
        - 'tx_loss_inr': value of energy lost to transmission
        - 'total_transaction_costs_inr': sum of all transaction costs
    """
    eta_discharge = bess_dict.get("eta_discharge", bess_dict.get("eta", 0.9220))
    fee_per_mwh = (
        costs_dict.get("iex_transaction_fee_inr_mwh", 20.0)
        + costs_dict.get("sldc_charge_inr_mwh", 5.0)
        + costs_dict.get("rldc_charge_inr_mwh", 2.0)
    )
    tx_loss_pct = costs_dict.get("transmission_loss_pct", 0.03)
    
    # IEX fees
    total_energy_traded = np.sum(P_dis * dt) + np.sum(P_ch * dt)
    iex_fees = total_energy_traded * costs_dict.get("iex_transaction_fee_inr_mwh", 20.0)
    
    # SLDC + RLDC
    sldc_rldc = total_energy_traded * (
        costs_dict.get("sldc_charge_inr_mwh", 5.0) + costs_dict.get("rldc_charge_inr_mwh", 2.0)
    )
    
    # Transmission loss value
    effective_discharge = np.sum(P_dis * eta_discharge * dt)
    tx_loss_energy = effective_discharge * tx_loss_pct / (1 - tx_loss_pct)
    tx_loss_value = tx_loss_energy * np.mean(prices[P_dis > 0]) if np.any(P_dis > 0) else 0.0
    
    return {
        "iex_fees_inr": float(iex_fees),
        "sldc_rldc_fees_inr": float(sldc_rldc),
        "tx_loss_inr": float(tx_loss_value),
        "total_transaction_costs_inr": float(iex_fees + sldc_rldc + tx_loss_value),
    }
