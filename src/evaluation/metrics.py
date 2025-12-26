import numpy as np
from scipy import stats

def mae(predicted, observed):
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        predicted (np.array): Forecast values
        observed (np.array): Ground truth values
    
    Returns:
        float: MAE
    """
    return np.mean(np.abs(predicted - observed))

def hit_rate(predicted_cat, observed_cat):
    """
    Calculate Hit Rate (Accuracy) for multiclass classification.
    
    Args:
        predicted_cat (np.array): Predicted categories
        observed_cat (np.array): Observed categories
        
    Returns:
        float: Fraction correct (0.0 to 1.0)
    """
    return np.mean(predicted_cat == observed_cat)

def rmse(predicted, observed):
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        predicted (np.array): Forecast values
        observed (np.array): Ground truth values
    
    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean((predicted - observed) ** 2))

def acc(predicted_anomaly, observed_anomaly, climatology=None):
    """
    Calculate Anomaly Correlation Coefficient (ACC).
    Often calculated as the spatial correlation for each time step, 
    then averaged over time.
    
    Args:
        predicted_anomaly (np.array): Forecast anomaly (Pred - Clim)
        observed_anomaly (np.array): Observed anomaly (Obs - Clim)
        
    Returns:
        float: Spatial Correlation Coefficient
    """
    # Flatten spatial dimensions (Lat, Lon) -> (N,)
    pred_flat = predicted_anomaly.flatten()
    obs_flat = observed_anomaly.flatten()
    
    # Avoid correlation of constant fields (std=0)
    if np.std(pred_flat) == 0 or np.std(obs_flat) == 0:
        return 0.0
        
    correlation, _ = stats.pearsonr(pred_flat, obs_flat)
    return correlation

def heidke_skill_score(predicted_cat, observed_cat):
    """
    Calculate Heidke Skill Score (HSS) for multiclass classification.
    
    Args:
        predicted_cat (np.array): Predicted categories (e.g., -1, 0, 1)
        observed_cat (np.array): Observed categories
        
    Returns:
        float: HSS score (-inf to 1]
    """
    predicted = predicted_cat.flatten()
    observed = observed_cat.flatten()
    
    correct = np.sum(predicted == observed)
    total = len(predicted)
    
    # Random chance accuracy (assuming 3 equal classes for terciles)
    # For a more robust HSS, we calculate Expected Correct using marginal totals
    # But for standard terciles, 1/3 is often used as a baseline approximation
    # Let's do the rigorous calculation:
    
    categories = np.unique(np.concatenate([predicted, observed]))
    
    expected_correct = 0
    for cat in categories:
        pred_freq = np.sum(predicted == cat)
        obs_freq = np.sum(observed == cat)
        expected_correct += (pred_freq * obs_freq) / total
        
    if total == expected_correct:
        return 0.0
        
    hss = (correct - expected_correct) / (total - expected_correct)
    return hss

def compute_metrics(predicted, observed, climatology, tercile_edges):
    """
    Wrapper to compute all metrics for a single event.
    """
    # 1. Deterministic
    err_rmse = rmse(predicted, observed)
    
    # 2. Skill (Anomalies)
    pred_anom = predicted - climatology
    obs_anom = observed - climatology
    score_acc = acc(pred_anom, obs_anom)
    
    # 3. Categorical (Terciles)
    # Classify based on edges: -1 (Below), 0 (Normal), 1 (Above)
    def categorize(data, edges):
        lower, upper = edges
        cat = np.zeros_like(data, dtype=int)
        cat[data < lower] = -1
        cat[data > upper] = 1
        return cat
        
    pred_cat = categorize(predicted, tercile_edges)
    obs_cat = categorize(observed, tercile_edges)
    
    score_hss = heidke_skill_score(pred_cat, obs_cat)
    
    return {
        'RMSE': err_rmse,
        'ACC': score_acc,
        'HSS': score_hss
    }
