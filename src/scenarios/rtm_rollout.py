import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .utils import inverse_cdf, fix_quantile_crossing_single
from ..models.quantile_lgbm import QuantileLGBM

class RTMRolloutGenerator:
    """
    RTM Autoregressive Rollout Generator.
    """
    def __init__(self, quantile_models: Dict[float, QuantileLGBM], 
                 feature_columns: List[str], 
                 updatable_features: List[str], 
                 seed=42):
        self.quantile_models = quantile_models
        self.feature_columns = feature_columns
        self.updatable_features = updatable_features
        self.rng = np.random.default_rng(seed)

    def generate(self, starting_features: pd.Series, n_scenarios: int, n_steps: int, 
                 cqr_corrections: Dict[str, float] = None,
                 external_uniforms: np.ndarray = None) -> np.ndarray:
        """
        Generate rollout scenarios.
        """
        scenarios = np.zeros((n_scenarios, n_steps))
        
        # We process scenario by scenario loop for simplicity given state updates
        # Vectorizing strictly would require batching predict which is possible but complex state logic.
        # But Prompt says "Performance... 6 mins acceptable".
        # We can optimize slightly by batching prediction if needed, but per-scenario loop is easier to implement robustly.
        
        # Actually, iterating 200 scenarios x 24 steps = 4800 loops.
        # Inside each loop, we verify models.
        # Can we vectorize across scenarios?
        # State at step t is different for each scenario (price lags).
        # We can run step t for ALL 200 scenarios at once.
        # X shape (200, n_features).
        
        # Vectorized Approach over Scenarios
        # Initialize Current Features for all scenarios
        # shape (n_scenarios, n_features)
        
        # Prepare base features mapping
        # We can store features as a DataFrame (n_scenarios, n_cols)
        # Replicating starting_features n_scenarios times
        
        features_df = pd.DataFrame([starting_features.values] * n_scenarios, columns=starting_features.index)
        
        # Cast to float for performance where possible
        features_df = features_df.astype(float, errors='ignore')
        
        # Track history for lags
        # We need a buffer of past prices.
        # Initialize with known history if available? 
        # The prompt says: "mcp_lag_1h <- price_buffer[-1]".
        # We only really need the dynamically generated prices.
        # What if t=0? lag_1h is from starting_features.
        # What if t=1? lag_1h is prediction at t=0.
        
        # We'll maintain a price_buffer array (n_scenarios, n_steps)
        # to look back easily.
        
        # Calendar logic preparation
        start_hour = int(starting_features['target_hour']) # or delivery_start_ist hour?
        # RTM features have 'target_hour' (or 'hour'?). 
        # Check Phase 2A summary: "RTM ... grid/weather ... cal_"
        # RTM features likely have 'cal_hour'.
        start_cal_hour = int(starting_features['cal_hour'])
        start_cal_day = int(starting_features['cal_day_of_week'])
        
        # Rollout steps
        for step in range(n_steps):
            # 1. Prepare X
            X = features_df[self.feature_columns]
            
            # 2. Predict Quantiles
            q_preds_arr = {}
            for alpha, model in self.quantile_models.items():
                q_preds_arr[alpha] = model.predict(X) 
                # returns ndarray (n_scenarios,)
                
            # NEW: Apply CQR correction if provided
            if cqr_corrections is not None:
                for alpha in list(q_preds_arr.keys()):
                    q_col = f"q{int(alpha * 100)}"
                    delta = cqr_corrections.get(q_col, 0)
                    q_preds_arr[alpha] = np.maximum(0.0, q_preds_arr[alpha] - delta)
                    
            # 3. Sort Crossing (Vectorized)
            # Stack (n_scenarios, n_quantiles)
            q_levels = sorted(q_preds_arr.keys())
            stacked = np.stack([q_preds_arr[q] for q in q_levels], axis=1)
            stacked = np.sort(stacked, axis=1)
            # Dict of arrays
            q_vals_fixed = {q: stacked[:, i] for i, q in enumerate(q_levels)}
            
            # 4. Sample Uniforms
            if external_uniforms is not None:
                u = external_uniforms[:, step]
            else:
                u = self.rng.uniform(0, 1, size=n_scenarios)
            
            # 5. Inverse CDF
            # Creating dict of ARRAYS for inverse_cdf (which I implemented to handle vector u)
            sampled_prices = inverse_cdf(u, q_vals_fixed)
            sampled_prices = np.maximum(0.0, sampled_prices)
            
            # Store
            scenarios[:, step] = sampled_prices
            
            # 6. Update Features for Step t+1
            if step < n_steps - 1:
                # Update Lags
                # lag_1h comes from 'sampled_prices' (which is step t)
                if 'mcp_lag_1h' in self.feature_columns:
                    features_df['mcp_lag_1h'] = sampled_prices
                
                # lag_2h comes from step t-1
                if 'mcp_lag_2h' in self.feature_columns:
                    if step >= 1:
                        features_df['mcp_lag_2h'] = scenarios[:, step-1]
                    # Else keep original (t=0, lag_2h is fixed from history)
                    
                if 'mcp_lag_4h' in self.feature_columns:
                    if step >= 3:
                        features_df['mcp_lag_4h'] = scenarios[:, step-3]
                
                # Rolling stats (approx)
                # mcp_rolling_mean_24h <- old_mean + (new - old_mean)/24
                # Wait, "old_mean" refers to the mean from the PREVIOUS STEP.
                # "new" is sampled_prices.
                # But what leaves the window?
                # Sliding window of 24.
                # Strictly: Mean_t+1 = Mean_t + (Price_t - Price_t-24)/24
                # We don't have Price_t-24 for the first 24 steps easily (it's in history).
                # Prompt suggests: "mcp_rolling_mean_24h <- old_mean + (sampled_price - old_mean) / 24"
                # This approximates it as exponential moving average or assuming outgoing is mean.
                if 'mcp_rolling_mean_24h' in self.feature_columns:
                     features_df['mcp_rolling_mean_24h'] = features_df['mcp_rolling_mean_24h'] + (sampled_prices - features_df['mcp_rolling_mean_24h']) / 24.0
                
                # Calendar Update
                next_hour = (start_cal_hour + step + 1) % 24
                features_df['cal_hour'] = next_hour
                features_df['cal_hour_sin'] = np.sin(2 * np.pi * next_hour / 24)
                features_df['cal_hour_cos'] = np.cos(2 * np.pi * next_hour / 24)
                
                # Day update
                total_hours_passed = start_cal_hour + step + 1
                if total_hours_passed % 24 == 0:
                     # Midnight crossed
                     current_day = features_df['cal_day_of_week']
                     new_day = (current_day + 1) % 7
                     features_df['cal_day_of_week'] = new_day
                     # Update weekend
                     features_df['cal_is_weekend'] = new_day.isin([5, 6]).astype(int)
        
        return scenarios
