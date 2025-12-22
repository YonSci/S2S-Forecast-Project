"""
Normalization utilities for S2S forecasting.

This module handles:
1. Climatology calculation and removal (anomaly creation)
2. Z-score normalization (zero mean, unit variance)
"""
import numpy as np
import xarray as xr
import pickle
import os

class S2SNormalizer:
    """
    Compute and apply climatology removal + normalization for S2S data.
    """
    
    def __init__(self):
        self.climatology = {}  # Weekly climatology maps (Week, H, W)
        self.mean = {}         # Overall mean (after clim removal)
        self.std = {}          # Overall std (after clim removal)
        self.tercile_33 = {}   # 33rd percentile maps (Week, H, W)
        self.tercile_66 = {}   # 66th percentile maps (Week, H, W)
        self.fitted = False
        
    def fit(self, samples, variable_names):
        """
        Compute climatology, normalization, and tercile statistics.
        
        Args:
            samples: List of (input_ds, target_ds) xarray datasets
        """
        print("Computing spatial-weekly statistics (Climatology & Terciles)...")
        
        # Initialize storage
        raw_values = {} # Temporarily store values to compute percentiles
        for var in variable_names['input'] + variable_names['target']:
            self.climatology[var] = None 
            self.mean[var] = 0.0
            self.std[var] = 1.0
            self.tercile_33[var] = None
            self.tercile_66[var] = None
            raw_values[var] = {} # var -> week -> list of 2D maps
            
        week_counts = {}
        
        def get_week(ds):
            t = ds.time.values
            if hasattr(t, 'size') and t.size > 1: t = t[0]
            return int(xr.DataArray(t).dt.isocalendar().week)

        for input_ds, target_ds in samples:
            in_week = get_week(input_ds)
            out_week = get_week(target_ds)
            
            # Inputs (atmospheric lag predictors)
            for var in variable_names['input']:
                if var in input_ds:
                    data = input_ds[var]
                    if 'pressure_level' in data.dims:
                        data = data.sel(pressure_level=500 if var == 'z' else 850)
                    
                    val = np.nan_to_num(data.values)
                    shape = val.shape
                    
                    if self.climatology[var] is None:
                        self.climatology[var] = np.zeros((54, *shape))
                        week_counts[var] = np.zeros(54)
                        for w in range(1, 54): raw_values[var][w] = []
                    
                    self.climatology[var][in_week] += val
                    week_counts[var][in_week] += 1
                    raw_values[var][in_week].append(val)
            
            # Targets (precipitation lead)
            for var in variable_names['target']:
                if var in target_ds:
                    val = np.nan_to_num(target_ds[var].values)
                    shape = val.shape
                    
                    if self.climatology[var] is None:
                        self.climatology[var] = np.zeros((54, *shape))
                        week_counts[var] = np.zeros(54)
                        for w in range(1, 54): raw_values[var][w] = []
                        
                    self.climatology[var][out_week] += val
                    week_counts[var][out_week] += 1
                    raw_values[var][out_week].append(val)
        
        # 1. Average climatology and compute Terciles
        for var in self.climatology.keys():
            shape = self.climatology[var].shape[1:]
            self.tercile_33[var] = np.zeros((54, *shape))
            self.tercile_66[var] = np.zeros((54, *shape))
            
            for w in range(1, 54):
                if week_counts[var][w] > 0:
                    self.climatology[var][w] /= week_counts[var][w]
                    
                    # Compute Terciles pixel-wise for this week
                    if raw_values[var][w]:
                        week_data = np.stack(raw_values[var][w], axis=0)
                        self.tercile_33[var][w] = np.percentile(week_data, 33.3, axis=0)
                        self.tercile_66[var][w] = np.percentile(week_data, 66.6, axis=0)
        
        # 2. Compute Standard Deviation of Anomalies
        print("Computing standard deviation of anomalies...")
        anomaly_sq_diffs = {var: [] for var in self.climatology.keys()}
        
        for input_ds, target_ds in samples:
            in_week = get_week(input_ds)
            out_week = get_week(target_ds)
            
            for var in variable_names['input']:
                if var in input_ds:
                    data = input_ds[var]
                    if 'pressure_level' in data.dims:
                        data = data.sel(pressure_level=500 if var == 'z' else 850)
                    anomaly = data.values - self.climatology[var][in_week]
                    anomaly_sq_diffs[var].append(anomaly.flatten())
            
            for var in variable_names['target']:
                if var in target_ds:
                    anomaly = target_ds[var].values - self.climatology[var][out_week]
                    anomaly_sq_diffs[var].append(anomaly.flatten())
                    
        for var in anomaly_sq_diffs.keys():
            if anomaly_sq_diffs[var]:
                all_anomalies = np.concatenate(anomaly_sq_diffs[var])
                all_anomalies = all_anomalies[~np.isnan(all_anomalies)]
                self.std[var] = np.nanstd(all_anomalies) or 1.0
                print(f"  {var}: std_anomaly={self.std[var]:.2f}")
        
        self.fitted = True

    
    def transform_input(self, input_ds, variable_names):
        """Apply normalization to input data."""
        if not self.fitted: raise ValueError("Not fitted")
        
        time_val = input_ds.time.values
        if isinstance(time_val, np.ndarray): time_val = time_val[0]
        week = int(xr.DataArray(time_val).dt.isocalendar().week)
        
        arrays = []
        for var in variable_names:
            if var in input_ds:
                data = input_ds[var]
                if 'pressure_level' in data.dims:
                    data = data.sel(pressure_level=500 if var == 'z' else 850)
                
                # Anomaly = Raw - Climatology[week]
                values = np.nan_to_num(data.values) - self.climatology[var][week]
                # Normalize
                values = values / (self.std[var] + 1e-8)
                arrays.append(values)
        
        return np.stack(arrays, axis=0)
    
    def transform_target(self, target_ds, variable_names):
        """Apply normalization to target data."""
        if not self.fitted: raise ValueError("Not fitted")
        
        time_val = target_ds.time.values
        if isinstance(time_val, np.ndarray): time_val = time_val[0]
        week = int(xr.DataArray(time_val).dt.isocalendar().week)
        
        arrays = []
        for var in variable_names:
            if var in target_ds:
                values = np.nan_to_num(target_ds[var].values) - self.climatology[var][week]
                values = values / (self.std[var] + 1e-8)
                arrays.append(values)
        
        return np.stack(arrays, axis=0)
    
    def inverse_transform_target(self, normalized_values, week, variable_name='precip'):
        """Convert normalized anomalies back to physical values using correct week's climatology."""
        # normalized_values shape: (H, W) or (1, H, W)
        if normalized_values.ndim == 3:
            normalized_values = normalized_values[0]
            
        # 1. De-normalize
        anomaly = normalized_values * self.std[variable_name]
        
        # 2. Add spatial climatology for that specific week
        physical = anomaly + self.climatology[variable_name][week]
        
        # Rainfall cannot be negative
        return np.maximum(physical, 0.0)
    
    def get_tercile_category(self, physical_values, week, variable_name='precip'):
        """
        Compare physical values to historical terciles and return:
        -1: Below Normal
         0: Near Normal
         1: Above Normal
        """
        if not self.fitted: raise ValueError("Not fitted")
        
        t33 = self.tercile_33[variable_name][week]
        t66 = self.tercile_66[variable_name][week]
        
        categories = np.zeros_like(physical_values)
        categories[physical_values < t33] = -1
        categories[physical_values >= t66] = 1
        
        return categories

    def save(self, filepath):
        """Save normalization statistics."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'climatology': self.climatology,
                'mean': self.mean,
                'std': self.std,
                'tercile_33': self.tercile_33,
                'tercile_66': self.tercile_66,
                'fitted': self.fitted
            }, f)

        print(f"Saved normalization statistics to {filepath}")
    
    def load(self, filepath):
        """Load normalization statistics from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.climatology = data['climatology']
            self.mean = data['mean']
            self.std = data['std']
            self.tercile_33 = data.get('tercile_33', {})
            self.tercile_66 = data.get('tercile_66', {})
            self.fitted = data['fitted']
        print(f"Loaded normalization statistics from {filepath}")

