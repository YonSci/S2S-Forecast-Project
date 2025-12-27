import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet import UNetGenerator
from src.data.dataloader import S2SDataset
from src.data.normalization import S2SNormalizer
from src.evaluation.metrics import compute_metrics, mae, hit_rate
# Climatology logic is embedded in standardization; Persistence logic implemented inline
# from src.evaluation.baselines import ClimatologyBaseline 

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

import cartopy.io.shapereader as shpreader
from matplotlib.path import Path

def get_ethiopia_mask(lats, lons):
    """Creates a boolean mask for Ethiopia vs Non-Ethiopia points."""
    try:
        shp_path = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shp_path)
        ethiopia_geom = None
        for record in reader.records():
            if record.attributes['NAME'] == 'Ethiopia':
                ethiopia_geom = record.geometry
                break
        
        if ethiopia_geom is None:
            print("Warning: Could not find Ethiopia shapefile. Skipping clipping.")
            return np.ones((len(lats), len(lons)), dtype=bool)
        
        # Create a meshgrid of points
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
        
        # Use simple bounds check first for speed
        min_lon, min_lat, max_lon, max_lat = ethiopia_geom.bounds
        
        # Detailed check using matplotlib Path
        # Handle MultiPolygon if necessary (Ethiopia is usually a single Polygon at 110m but safe to iterate)
        mask_flat = np.zeros(len(points), dtype=bool)
        
        geoms = ethiopia_geom.geoms if hasattr(ethiopia_geom, 'geoms') else [ethiopia_geom]
        
        for geom in geoms:
            # shell
            poly_path = Path(list(geom.exterior.coords))
            mask_flat |= poly_path.contains_points(points)
            # holes? (usually negligible for this resolution/purpose)
            
        return mask_flat.reshape(len(lats), len(lons))
        
    except Exception as e:
        print(f"Warning: Mask creation failed: {e}")
        return np.ones((len(lats), len(lons)), dtype=bool)


def plot_temporal_metrics(df, output_dir, suffix=""):
    """Generates time-series plots for metrics."""
    import matplotlib.dates as mdates
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Common plotting parameters
    def setup_axis(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Tick every 1 month
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        ax.grid(True, alpha=0.3)
    
    # 1. Error Metrics (RMSE, MAE)
    fig, ax = plt.subplots(figsize=(14, 7)) # Slightly wider
    ax.plot(df['Date'], df['RMSE_Model'], label='RMSE (Model)', color='#d62728', marker='o', linestyle='-', markersize=4)
    ax.plot(df['Date'], df['MAE_Model'], label='MAE (Model)', color='#ff7f0e', marker='s', linestyle='--', markersize=4)
    
    ax.set_title(f'Temporal Error Analysis (RMSE & MAE) - {suffix}', fontsize=16)
    ax.set_ylabel('Error (mm/day)', fontsize=12)
    ax.legend(fontsize=12)
    
    setup_axis(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_error_{suffix}.png'), dpi=300)
    plt.close()
    
    # 2. Score Metrics (ACC, HitRate, HSS)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Date'], df['ACC_Model'], label='ACC', color='#1f77b4', marker='o', markersize=4)
    ax.plot(df['Date'], df['HitRate_Model'], label='Hit Rate', color='#2ca02c', marker='^', markersize=4)
    
    ax.set_title(f'Temporal Skill Analysis (ACC & Hit Rate) - {suffix}', fontsize=16)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.legend(fontsize=12)
    
    setup_axis(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_skill_{suffix}.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(start_p_cat, start_t_cat, output_dir, suffix=""):
    """Generates a 3x3 confusion matrix for terciles."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Flatten arrays
    # Categories: -1 (Below), 0 (Normal), 1 (Above)
    # We want them sorted: Below, Normal, Above
    y_true = start_t_cat.flatten()
    y_pred = start_p_cat.flatten()
    
    labels = [-1, 0, 1]
    classes = ['Below', 'Normal', 'Above']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true') # Normalize by true rows
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
    
    plt.title(f'Tercile Confusion Matrix (Prob. given Observation) - {suffix}', fontsize=14)
    plt.ylabel('Observed Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{suffix}.png'), dpi=300)
    plt.close()
    
def plot_extreme_confusion_matrix(start_p_cat, start_t_cat, output_dir, suffix=""):
    """Generates a 2x2 confusion matrix for Extreme Events (Top 10%)."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_true = start_t_cat.flatten()
    y_pred = start_p_cat.flatten()
    
    labels = [0, 1]
    classes = ['Normal', 'Extreme']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true') 
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Reds', xticklabels=classes, yticklabels=classes, cbar=True)
    
    plt.title(f'Extreme Event (Top 10%) Matrix - {suffix}', fontsize=14)
    plt.ylabel('Observed', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'extreme_confusion_matrix_{suffix}.png'), dpi=300)
    plt.close()

def plot_scatter_metrics(p_mean, t_mean, output_dir, suffix=""):
    """Generates a scatter plot of Predicted vs Observed Domain-Averaged Rainfall."""
    from scipy.stats import pearsonr
    
    # Calculate correlation for annotation
    corr, _ = pearsonr(p_mean, t_mean)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter points
    ax.scatter(t_mean, p_mean, color='#1f77b4', alpha=0.6, edgecolors='b', s=50, label='Forecast')
    
    # 1:1 Line
    min_val = min(np.min(p_mean), np.min(t_mean))
    max_val = max(np.max(p_mean), np.max(t_mean))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Forecast')
    
    ax.set_title(f'Forecast vs Observation (Domain Avg) - {suffix}\nR = {corr[0]:.3f}', fontsize=16)
    ax.set_xlabel('Observed Mean Rainfall (mm/day)', fontsize=14)
    ax.set_ylabel('Forecast Mean Rainfall (mm/day)', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Force square aspect
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_plot_{suffix}.png'), dpi=300)
    plt.close()

def plot_spatial_metrics(bias_grid, rmse_grid, acc_grid, hitrate_grid, output_dir, suffix=""):
    """Generates and saves spacial bias, RMSE, ACC, and HitRate maps."""
    
    # Define coords (matching predict.py logic)
    lats = np.linspace(3, 15, 48)
    lons = np.linspace(33, 48, 60)
    
    # Create Mask
    mask = get_ethiopia_mask(lats, lons)
    
    # Apply Mask (Set False to NaN)
    # xarray where: cond=True (keep), False (replace)
    def mask_array(arr):
        return np.where(mask, arr, np.nan)
    
    ds = xr.Dataset(
        {
            'bias': (['lat', 'lon'], mask_array(bias_grid.squeeze())),
            'rmse': (['lat', 'lon'], mask_array(rmse_grid.squeeze())),
            'acc': (['lat', 'lon'], mask_array(acc_grid.squeeze())),
            'hitrate': (['lat', 'lon'], mask_array(hitrate_grid.squeeze()))
        },
        coords={'lat': lats, 'lon': lons}
    )
    
    common_cbar_kwargs = {'shrink': 0.7, 'pad': 0.05, 'aspect': 20}
    extent = [32.5, 48.5, 3, 15]
    
    def plot_map(var_name, cmap, levels, title, label, filename, extend='neither'):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)
        
        # Discrete Plot
        ds[var_name].plot(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap, 
            levels=levels,
            extend=extend,
            cbar_kwargs={**common_cbar_kwargs, 'label': label}
        )
        
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        plt.title(title, fontsize=14, pad=10)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # 1. Bias (Diverging, centered at 0)
    bias_levels = [-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
    plot_map('bias', 'RdBu', bias_levels, f"Spatial Bias (Forecast - Obs) - {suffix}", 'Mean Bias (mm/day)', f'spatial_bias_{suffix}.png', extend='both')

    # 2. RMSE (Sequential)
    rmse_levels = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    plot_map('rmse', 'viridis', rmse_levels, f"Spatial Root Mean Square Error - {suffix}", 'RMSE (mm/day)', f'spatial_rmse_{suffix}.png', extend='max')
    
    # 3. ACC (Diverging - Red High)
    acc_levels = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plot_map('acc', 'RdBu_r', acc_levels, f"Spatial Anomaly Correlation Coefficient - {suffix}", 'ACC', f'spatial_acc_{suffix}.png', extend='neither')
    
    # 4. Hit Rate (Sequential)
    hr_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plot_map('hitrate', 'plasma', hr_levels, f"Spatial Hit Rate (Tercile Accuracy) - {suffix}", 'Hit Rate', f'spatial_hitrate_{suffix}.png', extend='neither')


def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    
    # 1. Load Normalizer
    normalizer = S2SNormalizer()
    if not os.path.exists(config['normalizer_path']):
        raise FileNotFoundError(f"Normalizer not found at {config['normalizer_path']}")
    normalizer.load(config['normalizer_path'])
    
    # 3. Model
    model = UNetGenerator(input_channels=5, output_channels=1, target_size=(48, 60)).to(device)
    try:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {config['model_path']}")
        return
        
    model.eval()
    
    # 4. Tercile Edges
    TERCILE_EDGES = [-0.43, 0.43]
    
    print(f"Model Loaded. Evaluating individual years: {config['test_years']}")
    
    # Loop over each year to generate independent reports
    for year in config['test_years']:
        print(f"\nProcessing Year: {year}")
        
        # 2. Test Dataset (for single year)
        # Note: S2SDataset constructor accepts a list of years
        test_dataset = S2SDataset(
            data_dir=config['data_dir'],
            years=[year], 
            mode='test',
            normalizer=normalizer,
            lead_weeks=config['lead_weeks']
        )
        
        if len(test_dataset) == 0:
            print(f"No samples for {year}, skipping...")
            continue

        results = []
        spatial_preds = []
        spatial_targets = []
        
        all_targets = []
        for i in range(len(test_dataset)):
            _, target = test_dataset[i]
            all_targets.append(target.numpy())
        
        all_targets = np.array(all_targets) # (N, H, W)

        for i in range(len(test_dataset)):
            target_ds_raw = test_dataset.samples[i][1]
            sample_date = target_ds_raw.time.values
            
            input_tensor, target_tensor = test_dataset[i]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            target_np = target_tensor.numpy()
            
            # A. Model Predict
            with torch.no_grad():
                pred_tensor = model(input_tensor).cpu().numpy()[0]
                
            # Collect for spatial
            spatial_preds.append(pred_tensor * normalizer.std['precip'].astype(np.float32))
            spatial_targets.append(target_np * normalizer.std['precip'].astype(np.float32))
                
            # B. Baselines
            clim_pred = np.zeros_like(target_np)
            
            pidx = i - config['lead_weeks']
            if pidx >= 0:
                pers_pred = all_targets[pidx]
            else:
                pers_pred = np.zeros_like(target_np)
                
            # Compute Metrics
            def get_scores(prediction, truth, label):
                metrics = compute_metrics(prediction, truth, 0.0, TERCILE_EDGES) 
                metrics['MAE'] = mae(prediction, truth)
                def cat(d):
                    c = np.zeros_like(d, dtype=int)
                    c[d < TERCILE_EDGES[0]] = -1
                    c[d > TERCILE_EDGES[1]] = 1
                    return c
                p_cat = cat(prediction)
                t_cat = cat(truth)
                metrics['HitRate'] = hit_rate(p_cat, t_cat)
                return {f"{k}_{label}": v for k, v in metrics.items()}

            scores_model = get_scores(pred_tensor, target_np, "Model")
            scores_clim = get_scores(clim_pred, target_np, "Clim")
            scores_pers = get_scores(pers_pred, target_np, "Pers")
            
            results.append({
                'Date': sample_date,
                **scores_model, 
                **scores_clim, 
                **scores_pers
            })

        # Aggregation
        df = pd.DataFrame(results)
        df = df.iloc[config['lead_weeks']:] # Skip first weeks for persistence
        
        # --- Spatial Analysis ---
        sp_p = np.array(spatial_preds)[config['lead_weeks']:]
        sp_t = np.array(spatial_targets)[config['lead_weeks']:]
        
        bias_map = np.mean(sp_p - sp_t, axis=0)
        mse_map = np.mean((sp_p - sp_t)**2, axis=0)
        rmse_map = np.sqrt(mse_map)
        
        p_anom = sp_p - np.mean(sp_p, axis=0)
        t_anom = sp_t - np.mean(sp_t, axis=0)
        numerator = np.sum(p_anom * t_anom, axis=0)
        denominator = np.sqrt(np.sum(p_anom**2, axis=0) * np.sum(t_anom**2, axis=0))
        acc_map = np.zeros_like(bias_map)
        valid_mask = denominator != 0
        acc_map[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        lower_tercile = np.percentile(sp_t, 33.333, axis=0)
        upper_tercile = np.percentile(sp_t, 66.666, axis=0)
        def discretize(data):
            cats = np.zeros_like(data, dtype=int)
            cats[data < lower_tercile] = -1 
            cats[data > upper_tercile] = 1 
            return cats
        p_cat = discretize(sp_p)
        t_cat = discretize(sp_t)
        hitrate_map = np.mean((p_cat == t_cat).astype(float), axis=0)
        
        # Extreme Events (>90th Percentile)
        extreme_thresh = np.percentile(sp_t, 90, axis=0) # Per pixel threshold
        p_ext = (sp_p > extreme_thresh).astype(int)
        t_ext = (sp_t > extreme_thresh).astype(int)
        
        print(f"Saving Spatial Maps ({year})...")
        plot_spatial_metrics(bias_map, rmse_map, acc_map, hitrate_map, "outputs", suffix=str(year))
        
        # --- Temporal Analysis ---
        print(f"Saving Temporal Plots ({year})...")
        plot_temporal_metrics(df, "outputs", suffix=str(year))
        
        # --- Confusion Matrices ---
        print(f"Saving Confusion Matrices ({year})...")
        plot_confusion_matrix(p_cat, t_cat, "outputs", suffix=str(year))
        plot_extreme_confusion_matrix(p_ext, t_ext, "outputs", suffix=str(year))
        
        # --- Scatter Plot ---
        print(f"Saving Scatter Plot ({year})...")
        # sp_p shape is (N, H, W). Mean over H, W (axes 1, 2)
        domain_p = np.mean(sp_p, axis=(1, 2))
        domain_t = np.mean(sp_t, axis=(1, 2))
        plot_scatter_metrics(domain_p, domain_t, "outputs", suffix=str(year))
        
        # --- Report Generation ---
        report_data = {
            "meta": {
                "year": year,
                "lead_weeks": config['lead_weeks'],
                "samples": len(df)
            },
            "metrics": []
        }
        
        print("-" * 40)
        print(f"REPORT {year}")
        metrics_to_show = ["RMSE", "MAE", "ACC", "HSS", "HitRate"]
        for m in metrics_to_show:
            val_mod = df[f"{m}_Model"].mean()
            val_pers = df[f"{m}_Pers"].mean()
            val_clim = df[f"{m}_Clim"].mean()
            if m in ["RMSE", "MAE"]:
                skill = 1 - (val_mod / val_clim)
            else:
                skill = val_mod - val_clim
            
            print(f"{m:<8}: {val_mod:.4f} (Skill: {skill:.4f})")
            report_data["metrics"].append({
                "name": m,
                "model": float(val_mod),
                "persistence": float(val_pers),
                "climatology": float(val_clim),
                "skill": float(skill)
            })
            
        out_path = os.path.join("outputs", f"evaluation_report_{year}.json")
        import json
        with open(out_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved: {out_path}")
    
    print("\nBatch Evaluation Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, nargs='+', default=["2020"], help="Test years")
    parser.add_argument("--lead", type=int, default=1, help="Lead weeks")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    # Parse years
    years = []
    for y in args.years:
        if '-' in y:
            s, e = map(int, y.split('-'))
            years.extend(range(s, e+1))
        else:
            years.append(int(y))
            
    config = {
        'data_dir': 'data/train',
        'test_years': years,
        'lead_weeks': args.lead,
        'model_path': args.model,
        'normalizer_path': 'checkpoints/normalizer.pkl'
    }
    
    evaluate_model(config)
