import numpy as np
from typing import Dict, Union

def fix_quantile_crossing(predictions_dict: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
    """
    Enforce monotonicity: q10 <= q25 <= q50 <= q75 <= q90
    """
    quantiles = sorted(predictions_dict.keys())
    # Stack predictions: (n_samples, n_quantiles)
    stacked = np.stack([predictions_dict[q] for q in quantiles], axis=1)
    
    # Sort across quantiles
    stacked_sorted = np.sort(stacked, axis=1)
    
    # Unpack back to dictionary
    fixed_dict = {}
    for i, q in enumerate(quantiles):
        fixed_dict[q] = stacked_sorted[:, i]
        
    return fixed_dict

def fix_quantile_crossing_single(predictions_dict: Dict[float, float]) -> Dict[float, float]:
    """
    Enforce monotonicity for single sample prediction.
    """
    quantiles = sorted(predictions_dict.keys())
    values = [predictions_dict[q] for q in quantiles]
    values_sorted = sorted(values)
    return {q: v for q, v in zip(quantiles, values_sorted)}


def inverse_cdf(u: Union[float, np.ndarray], quantile_values: Dict[float, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
    """
    Piecewise linear interpolation of Inverse CDF.
    """
    q_levels = sorted(quantile_values.keys()) # [0.1, 0.25, 0.5, 0.75, 0.9]
    # Check if input is array
    is_array = isinstance(u, np.ndarray)
    
    # Prepare output
    if is_array:
        price = np.zeros_like(u)
    else:
        price = 0.0

    # Get Quantile Values
    q_vals = [quantile_values[q] for q in q_levels]

    # Helper for scalar logic
    def _inv_cdf_scalar(val, q_vals, q_levels):
        # 1. Extrapolate below q10
        if val < q_levels[0]:
            # Slope between q10 and q25
            slope = (q_vals[1] - q_vals[0]) / (q_levels[1] - q_levels[0])
            # Linear ext from q10
            est = q_vals[0] - slope * (q_levels[0] - val)
            # Floor
            floor_val = max(0, q_vals[0] - 2 * (q_vals[1] - q_vals[0]))
            # Assign CDF=0.01 roughly map to floor? 
            # Prompt says: "assign CDF = 0.01" to mean the value at u=0.01 is the floor?
            # Or just "assign CDF=0.01" is cryptic in the prompt.
            # "Extrapolate below q10: extend from q10-q25 slope, floor at max(0, q10 - 2*(q25 - q10)), assign CDF = 0.01"
            # This implies the floor is reached at u=0.01? Or 0.0?
            # Let's assume standard linear extrapolation but clamped.
            
            # Re-read prompt carefully:
            # "floor at max(0, q10 - 2*(q25 - q10))... assign CDF = 0.01"
            # Maybe it means the floor corresponds to u=0.01.
            # Let's effectively interpolate between (0.01, floor) and (0.10, q10).
            floor_p = max(0, q_vals[0] - 2 * (q_vals[1] - q_vals[0]))
            
            if val < 0.01:
                return floor_p
            else:
                # Interp between 0.01 and 0.10
                ratio = (val - 0.01) / (q_levels[0] - 0.01)
                return floor_p + ratio * (q_vals[0] - floor_p)
                
        # 2. Extrapolate above q90
        elif val > q_levels[-1]:
            # "cap at q90 + 2*(q90 - q75), assign CDF = 0.99"
            cap_p = q_vals[-1] + 2 * (q_vals[-1] - q_vals[-2])
            
            if val > 0.99:
                return cap_p
            else:
                 # Interp between 0.90 and 0.99
                ratio = (val - q_levels[-1]) / (0.99 - q_levels[-1])
                return q_vals[-1] + ratio * (cap_p - q_vals[-1])
        
        # 3. Interpolate between known points
        else:
            return np.interp(val, q_levels, q_vals)

    if is_array:
        # Check if quantile values are arrays (one per scenario)
        if isinstance(q_vals[0], np.ndarray):
            return np.array([_inv_cdf_scalar(u[i], [qv[i] for qv in q_vals], q_levels) for i in range(len(u))])
        else:
            # Case where we have one quantile set but multiple u samples
            return np.array([_inv_cdf_scalar(v, q_vals, q_levels) for v in u])
    else:
        return _inv_cdf_scalar(u, q_vals, q_levels)
