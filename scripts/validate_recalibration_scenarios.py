import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate_recalibration_scenarios():
    print("=" * 64)
    print("VALIDATING RECALIBRATED SCENARIO COVERAGE")
    print("=" * 64)
    
    # Load actuals
    # Note: Using the same actuals as the sweep
    dam_actuals = pd.read_csv("Data/Predictions/dam_quantiles_backtest.csv")
    rtm_actuals = pd.read_csv("Data/Predictions/rtm_quantiles_backtest.csv")
    
    # Load scenarios
    dam_scen = pd.read_parquet("Data/Predictions/joint_dam_scenarios_backtest_recalibrated.parquet")
    rtm_scen = pd.read_parquet("Data/Predictions/joint_rtm_scenarios_backtest_recalibrated.parquet")
    
    h_cols = [f"h{h:02d}" for h in range(24)]
    
    results = []
    for market, act_df, scen_df in [("DAM", dam_actuals, dam_scen), ("RTM", rtm_actuals, rtm_scen)]:
        # Melt scenarios to long
        scen_long = scen_df.melt(id_vars=['target_date', 'scenario_id'], value_vars=h_cols, var_name='hour_str', value_name='price')
        scen_long['target_hour'] = scen_long['hour_str'].str.extract(r'(\d+)').astype(int)
        
        # Merge with actuals
        # Actuals column is 'target_mcp_rs_mwh'
        act_col = 'target_mcp_rs_mwh'
        merged = pd.merge(scen_long, act_df[['target_date', 'target_hour', act_col]], on=['target_date', 'target_hour'])
        
        # Coverage per quantile
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
            # Compute scenario-based quantile per (date, hour)
            day_hour_qs = scen_long.groupby(['target_date', 'target_hour'])['price'].quantile(alpha).reset_index()
            day_hour_qs.columns = ['target_date', 'target_hour', f'q_scen_{int(alpha*100)}']
            
            # Merge with actuals for this alpha
            coverage_df = pd.merge(day_hour_qs, act_df[['target_date', 'target_hour', act_col]], on=['target_date', 'target_hour'])
            cov = (coverage_df[act_col] <= coverage_df[f'q_scen_{int(alpha*100)}']).mean()
            
            results.append({
                "Market": market,
                "Alpha": alpha,
                "Coverage": cov,
                "Error": cov - alpha
            })
            
    res_df = pd.DataFrame(results)
    print(res_df.pivot(index='Alpha', columns='Market', values=['Coverage', 'Error']))
    
if __name__ == "__main__":
    validate_recalibration_scenarios()
