import pandas as pd
import numpy as np
import warnings
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

    def get_multiday_scenarios(self, start_date: str, n_days: int = 7,
                                n_scenarios: int = 100) -> Dict[str, np.ndarray]:
        """
        Returns multi-day DAM and RTM scenarios starting from start_date.
        
        Attempts to load from pre-built multiday scenario parquets.
        Falls back to single-day loading for Day 0 with RuntimeWarning.
        
        Returns:
            {
                'dam': np.ndarray (n_scenarios, n_days, 24),
                'rtm': np.ndarray (n_scenarios, n_days, 24),
                'dam_actual': np.ndarray (24,)  — Day D only
                'rtm_actual': np.ndarray (24,)  — Day D only
            }
        """
        # Try multiday parquets
        multiday_dam_path = self.dam_path.parent / "multiday_dam_scenarios_backtest.parquet"
        multiday_rtm_path = self.dam_path.parent / "multiday_rtm_scenarios_backtest.parquet"
        
        h_cols = [f'h{i:02d}' for i in range(24)]
        
        if multiday_dam_path.exists() and multiday_rtm_path.exists():
            # Lazy load multiday data
            if not hasattr(self, '_multiday_dam_df'):
                self._multiday_dam_df = pd.read_parquet(multiday_dam_path)
                self._multiday_rtm_df = pd.read_parquet(multiday_rtm_path)
            
            dam_3d = np.zeros((n_scenarios, n_days, 24))
            rtm_3d = np.zeros((n_scenarios, n_days, 24))
            
            for d in range(n_days):
                mask_d = (
                    (self._multiday_dam_df['target_date'] == start_date) 
                    & (self._multiday_dam_df['day_offset'] == d)
                )
                mask_r = (
                    (self._multiday_rtm_df['target_date'] == start_date) 
                    & (self._multiday_rtm_df['day_offset'] == d)
                )
                
                d_day = self._multiday_dam_df[mask_d].sort_values('scenario_id')
                r_day = self._multiday_rtm_df[mask_r].sort_values('scenario_id')
                
                n_avail = min(len(d_day), n_scenarios)
                if n_avail > 0:
                    dam_3d[:n_avail, d, :] = d_day.iloc[:n_avail][h_cols].values
                    rtm_3d[:n_avail, d, :] = r_day.iloc[:n_avail][h_cols].values
                    # Fill extras by resampling
                    if n_avail < n_scenarios:
                        for i in range(n_avail, n_scenarios):
                            idx = i % n_avail
                            dam_3d[i, d, :] = dam_3d[idx, d, :]
                            rtm_3d[i, d, :] = rtm_3d[idx, d, :]
        else:
            # Fallback: only Day 0 from per-day scenarios — DEGENERATE lookahead
            warnings.warn(
                f"Multiday parquets not found at {multiday_dam_path}. "
                f"Falling back to Day 0 repetition for {start_date} — "
                f"multi-day lookahead is degenerate (Days 1-{n_days-1} = copy of Day 0). "
                f"Run build_multiday_scenarios.py first for meaningful cross-day optimization.",
                RuntimeWarning,
                stacklevel=2
            )
            day_data = self.get_day_scenarios(start_date, n_scenarios)
            dam_3d = np.zeros((n_scenarios, n_days, 24))
            rtm_3d = np.zeros((n_scenarios, n_days, 24))
            dam_3d[:, 0, :] = day_data['dam'][:n_scenarios]
            rtm_3d[:, 0, :] = day_data['rtm'][:n_scenarios]
            # Days 1+ just repeat Day 0 (no cross-day info)
            for d in range(1, n_days):
                dam_3d[:, d, :] = dam_3d[:, 0, :]
                rtm_3d[:, d, :] = rtm_3d[:, 0, :]
        
        # Actuals for Day D (load once via single-day loader)
        day_data = self.get_day_scenarios(start_date, n_scenarios)
        
        return {
            'dam': dam_3d,
            'rtm': rtm_3d,
            'dam_actual': day_data['dam_actual'],
            'rtm_actual': day_data['rtm_actual']
        }
