import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

class ScenarioLoader:
    """
    Efficiently load joint DAM/RTM scenarios and actuals for Phase 3B.
    """
    def __init__(self, dam_path: str, rtm_path: str, actuals_dam_path: str, actuals_rtm_path: str):
        self.dam_path = Path(dam_path)
        self.rtm_path = Path(rtm_path)
        self.actuals_dam_path = Path(actuals_dam_path)
        self.actuals_rtm_path = Path(actuals_rtm_path)
        
        # Load everything into memory once
        print(f"Loading scenarios from {self.dam_path}...")
        self.dam_df = pd.read_parquet(self.dam_path)
        print(f"Loading scenarios from {self.rtm_path}...")
        self.rtm_df = pd.read_parquet(self.rtm_path)
        print(f"Loading DAM actuals from {self.actuals_dam_path}...")
        self.actuals_dam_df = pd.read_csv(self.actuals_dam_path)
        print(f"Loading RTM actuals from {self.actuals_rtm_path}...")
        self.actuals_rtm_df = pd.read_csv(self.actuals_rtm_path)
        
        # Pre-process dates
        self.common_dates = sorted(list(set(self.dam_df['target_date'].unique())))
        
    def get_day_scenarios(self, target_date: str, n_scenarios: int = 100) -> Dict[str, np.ndarray]:
        """
        Returns DAM and RTM scenarios for a specific day.
        Returns:
            {
                'dam': np.ndarray (n_scenarios, 24),
                'rtm': np.ndarray (n_scenarios, 24),
                'dam_actual': np.ndarray (24,)
            }
        """
        # Filter scenarios
        d_day = self.dam_df[self.dam_df['target_date'] == target_date].sort_values('scenario_id')
        r_day = self.rtm_df[self.rtm_df['target_date'] == target_date].sort_values('scenario_id')
        
        # Limit to n_scenarios
        if len(d_day) > n_scenarios * 1: # Grouped by scenario_id
            # Wait, dam_df is already wide format h00..h23
            d_day = d_day.iloc[:n_scenarios]
            r_day = r_day.iloc[:n_scenarios]
            
        # Extract matrices h00..h23
        h_cols = [f'h{i:02d}' for i in range(24)]
        dam_scen = d_day[h_cols].values
        rtm_scen = r_day[h_cols].values
        
        # Actuals
        a_day_d = self.actuals_dam_df[self.actuals_dam_df['target_date'] == target_date].sort_values('target_hour')
        actual_col_d = 'actual_mcp' if 'actual_mcp' in a_day_d.columns else 'target_mcp_rs_mwh'
        dam_actual = a_day_d[actual_col_d].values
        
        a_day_r = self.actuals_rtm_df[self.actuals_rtm_df['target_date'] == target_date].sort_values('target_hour')
        actual_col_r = 'actual_mcp' if 'actual_mcp' in a_day_r.columns else 'target_mcp_rs_mwh'
        rtm_actual = a_day_r[actual_col_r].values
        
        return {
            'dam': dam_scen,
            'rtm': rtm_scen,
            'dam_actual': dam_actual,
            'rtm_actual': rtm_actual
        }
