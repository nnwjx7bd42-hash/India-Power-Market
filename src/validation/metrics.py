"""
Evaluation metrics for price forecasting models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """
    Calculate Mean Absolute Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    epsilon : float
        Small value to avoid division by zero
    
    Returns:
    --------
    float
        MAPE as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def hit_rate(y_true, y_pred, atol=1e-9):
    """
    Hit rate (perfect prediction rate): fraction of periods where forecast was exactly correct.

    Hit Rate = (Number of exact matches / Total forecast periods) × 100%

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    atol : float
        Absolute tolerance for "exact" match (default 1e-9; use 0 for strict equality)

    Returns:
    --------
    float
        Hit rate as percentage (0-100)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan
    exact = np.isclose(y_true, y_pred, rtol=0, atol=atol)
    return np.mean(exact) * 100


def accuracy_within_tolerance(y_true, y_pred, bands=(5, 10), epsilon=1e-8):
    """
    Percentage of forecasts falling within tolerance bands (e.g. ±5%, ±10%).

    - Within ±5%: strong accuracy
    - Within ±10%: acceptable accuracy
    - Beyond ±10%: miss

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    bands : tuple of float
        Tolerance percentages, e.g. (5, 10) for 5% and 10%
    epsilon : float
        Minimum |actual| to avoid division by zero

    Returns:
    --------
    dict
        Keys like "within_5pct", "within_10pct", "beyond_10pct" with percentages (0-100).
        Band names use the first band as "within_Xpct" and the last as "beyond_Xpct".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {f"within_{int(b)}pct": np.nan for b in sorted(bands)} | {
            f"beyond_{int(sorted(bands)[-1])}pct": np.nan
        }

    pct_error = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), epsilon) * 100
    out = {}
    for i, b in enumerate(sorted(bands)):
        key = f"within_{int(b)}pct"
        if i == 0:
            out[key] = np.mean(pct_error <= b) * 100
        else:
            prev_b = sorted(bands)[i - 1]
            out[key] = np.mean((pct_error > prev_b) & (pct_error <= b)) * 100
    out[f"beyond_{int(bands[-1])}pct"] = np.mean(pct_error > bands[-1]) * 100
    return out


def forecast_bias_counts(y_true, y_pred, atol=1e-9):
    """
    Count how many times forecast was above actual, below actual, or exact over the full period.

    Returns:
    --------
    dict
        n_total, n_over (forecast > actual), n_under (forecast < actual), n_exact (forecast ≈ actual)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"n_total": 0, "n_over": 0, "n_under": 0, "n_exact": 0}
    over = np.sum(y_pred > y_true)
    under = np.sum(y_pred < y_true)
    exact = np.sum(np.isclose(y_true, y_pred, rtol=0, atol=atol))
    return {"n_total": int(n), "n_over": int(over), "n_under": int(under), "n_exact": int(exact)}


def within_tolerance_bias(y_true, y_pred, tolerance_pct=5, epsilon=1e-8, atol=1e-9):
    """
    Among forecasts within ±tolerance_pct of actual: count over (forecast > actual),
    under (forecast < actual), and exact.

    Answers: "When the model was within ±5% of actual, how often did it say high vs low?"
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"n_within": 0, "n_over": 0, "n_under": 0, "n_exact": 0}
    pct_error = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), epsilon) * 100
    within_mask = pct_error <= tolerance_pct
    n_within = int(np.sum(within_mask))
    if n_within == 0:
        return {"n_within": 0, "n_over": 0, "n_under": 0, "n_exact": 0}
    y_t = y_true[within_mask]
    y_p = y_pred[within_mask]
    over = int(np.sum(y_p > y_t))
    under = int(np.sum(y_p < y_t))
    exact = int(np.sum(np.isclose(y_t, y_p, rtol=0, atol=atol)))
    return {"n_within": n_within, "n_over": over, "n_under": under, "n_exact": exact}


def calculate_mda(y_true, y_pred):
    """
    Mean Directional Accuracy (MDA): fraction of periods where the forecast correctly
    predicted the direction of change compared to the previous actual.

    MDA = (1/N) × Σ 𝟙[sgn(Forecast_t − Actual_{t−1}) = sgn(Actual_t − Actual_{t−1})] × 100%

    Valuable in energy/commodity markets where trend direction often matters more than magnitude.

    Parameters:
    -----------
    y_true : array-like
        Actual values (time-ordered)
    y_pred : array-like
        Predicted values (aligned with y_true)

    Returns:
    --------
    float
        MDA as percentage (0-100), or np.nan if fewer than 2 observations
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan
    n = min(len(y_true), len(y_pred))
    actual_prev = y_true[: n - 1]
    actual_curr = y_true[1:n]
    pred_curr = y_pred[1:n]
    dir_actual = np.sign(actual_curr - actual_prev)
    dir_forecast = np.sign(pred_curr - actual_prev)
    correct = dir_actual == dir_forecast
    return np.mean(correct) * 100


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy - percentage of correct price direction predictions
    (compares sign of actual change vs sign of predicted change).

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    float
        Directional accuracy as percentage (0-100)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate price changes
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)

    # Count correct directions (same sign)
    correct = np.sign(true_changes) == np.sign(pred_changes)

    if len(correct) == 0:
        return np.nan

    return np.mean(correct) * 100


def calculate_r2(y_true, y_pred):
    """Calculate R² Score"""
    return r2_score(y_true, y_pred)


def forecast_metrics_report(y_true, y_pred, tolerance_pct=(5, 10), include_magnitude=True):
    """
    Combined forecast evaluation: hit rate, tolerance bands, MDA, and magnitude metrics.

    Captures precision (exact matches), acceptable accuracy (±X% bands), directional
    reliability (MDA), and average deviation (MAPE, MAE).

    Parameters:
    -----------
    y_true : array-like
        Actual values (time-ordered)
    y_pred : array-like
        Predicted values
    tolerance_pct : tuple of float
        Tolerance bands in percent, e.g. (5, 10) for ±5% and ±10%
    include_magnitude : bool
        If True, include MAPE, MAE, RMSE, R2

    Returns:
    --------
    dict
        hit_rate_pct, within_5pct, within_10pct, beyond_10pct (from tolerance_pct),
        mda_pct, and optionally mape, mae, rmse, r2
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = {
        "hit_rate_pct": float(hit_rate(y_true, y_pred)),
        "mda_pct": float(calculate_mda(y_true, y_pred)),
    }
    tol = accuracy_within_tolerance(y_true, y_pred, bands=tolerance_pct)
    report.update({k: float(v) for k, v in tol.items()})
    # Full period: times forecast > actual vs < actual
    bias_full = forecast_bias_counts(y_true, y_pred)
    report["n_over_actual"] = bias_full["n_over"]
    report["n_under_actual"] = bias_full["n_under"]
    report["n_exact"] = bias_full["n_exact"]
    # Within ±first_tolerance (e.g. ±5%): over vs under
    first_band = min(tolerance_pct)
    bias_within = within_tolerance_bias(y_true, y_pred, tolerance_pct=first_band)
    report[f"within_{int(first_band)}pct_n_total"] = bias_within["n_within"]
    report[f"within_{int(first_band)}pct_n_over"] = bias_within["n_over"]
    report[f"within_{int(first_band)}pct_n_under"] = bias_within["n_under"]
    report[f"within_{int(first_band)}pct_n_exact"] = bias_within["n_exact"]
    if include_magnitude:
        report["mape"] = float(calculate_mape(y_true, y_pred))
        report["mae"] = float(calculate_mae(y_true, y_pred))
        report["rmse"] = float(calculate_rmse(y_true, y_pred))
        report["r2"] = float(calculate_r2(y_true, y_pred))
    return report


def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    dict
        Dictionary of metric names and values
    """
    metrics = {
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'Directional_Accuracy': calculate_directional_accuracy(y_true, y_pred)
    }

    return metrics


def calculate_metrics_by_group(y_true, y_pred, groups):
    """
    Calculate metrics grouped by a categorical variable
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    groups : array-like
        Group labels
    
    Returns:
    --------
    pd.DataFrame
        Metrics for each group
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': groups
    })
    
    results = []
    for group in df['group'].unique():
        mask = df['group'] == group
        group_metrics = calculate_metrics(
            df.loc[mask, 'y_true'],
            df.loc[mask, 'y_pred']
        )
        group_metrics['group'] = group
        group_metrics['n_samples'] = mask.sum()
        results.append(group_metrics)
    
    return pd.DataFrame(results)
