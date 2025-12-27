import torch
import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# --- CONFIGURATION ---
LEAD_WEEKS = 1  # The number of weeks ahead to forecast
# ---------------------

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet import UNetGenerator
from src.data.normalization import S2SNormalizer

def mask_to_ethiopia(ds):
    """
    Mask the xarray dataset to only show values within the Ethiopia boundary.
    """
    if not CARTOPY_AVAILABLE:
        return ds
        
    print("Applying Ethiopia boundary mask...")
    try:
        # Get Ethiopia geometry from Natural Earth
        shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        ethiopia_geom = [country.geometry for country in reader.records() if country.attributes['NAME'] == 'Ethiopia'][0]
        
        # Create a mask
        lats = ds.lat.values
        lons = ds.lon.values
        
        # Create a grid of points
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
        
        # Vectorized check using shapely
        mask = np.array([ethiopia_geom.contains(Point(p)) for p in points])
        mask = mask.reshape(len(lats), len(lons))
        
        # Apply mask
        ds = ds.where(xr.DataArray(mask, coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon']))
        
    except Exception as e:
        print(f"Warning: Could not apply Ethiopia mask: {e}")
        
    return ds

def visualize_forecast(ds, var_name, init_date, lead_date, lead_weeks, output_path):
    """
    Produce a professional-quality forecast map for Ethiopia.
    """
    # Force a clean scientific style with high-contrast labels
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.labelweight': 'bold',
        'font.family': 'sans-serif',
        'savefig.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    
    # 1. Colormap Selection
    plot_kwargs = {}
    
    if 'anomaly' in var_name:
        # Dynamic Discrete Levels
        # 1. Determine robustness range
        data_min = ds[var_name].min().item() if not np.isnan(ds[var_name].min()) else 0
        data_max = ds[var_name].max().item() if not np.isnan(ds[var_name].max()) else 0
        limit = max(abs(data_min), abs(data_max))
        if limit == 0: limit = 1.0
        
        # 2. Create 11 levels (10 bins) centered at 0
        levels = np.linspace(-limit, limit, 11)
        
        # 3. Custom RYG Colormap (Red -> Yellow -> White -> Green)
        colors = ['#d73027', '#fdae61', '#ffffbf', '#ffffff', '#d9ef8b', '#a6d96a', '#1a9850']
        n_bins = len(levels) - 1
        cmap = LinearSegmentedColormap.from_list('custom_ryg', colors, N=n_bins)
        
        label = 'Precipitation Anomaly (mm/day)'
        title_prefix = "Precipitation Anomaly"
        
        vmin, vmax = None, None # Handled by levels
        plot_kwargs = {'levels': levels, 'extend': 'both'}
        
    elif 'percent' in var_name:
        # Custom discrete colormap for Percent of Normal
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
        levels = np.linspace(0, 200, len(colors) + 1)
        cmap = LinearSegmentedColormap.from_list('percent_cmap', colors, N=len(colors))
        label = 'Percent of Normal Rainfall (%)'
        title_prefix = "Relative Precipitation"
        vmin, vmax = 0, 200
        plot_kwargs = {'levels': levels, 'extend': 'neither'}
    elif 'tercile' in var_name:
        from matplotlib.colors import ListedColormap
        # Discrete: Below (Orange/Red), Normal (Cyan), Above (Green)
        # Colors match user preference: Red -> Cyan -> Green
        cmap = ListedColormap(['#d73027', '#74add1', '#1a9850'])
        label = 'Tercile Category'
        title_prefix = "Tercile Forecast"
        vmin, vmax = -1.5, 1.5
    else:
        # Dynamic Discrete Levels for Total Precip
        data_max = ds[var_name].max().item() if not np.isnan(ds[var_name].max()) else 0
        if data_max == 0: data_max = 12.0 # Fallback
        
        # Create 10 levels from 0 to max
        levels = np.linspace(0, data_max, 10)
        
        # Custom Grey -> Orange -> Green Palette
        # Matches user reference: Grey (low) -> Orange -> Yellow -> Green (high)
        colors = [
            '#e0e0e0', # Grey (Lowest/Zero)
            '#fdbb84', # Orange
            '#fee08b', # Yellow-Orange
            '#ffffbf', # Pale Yellow
            '#d9f0a3', # Pale Green
            '#addd8e', # Light Green
            '#78c679', # Medium Green
            '#31a354', # Green
            '#006837'  # Dark Green (Highest)
        ]
        
        # Interpolate/Resample to match number of bins if needed, but here we just use the list
        # For discrete consistency, we treat these as N distinct colors
        cmap = LinearSegmentedColormap.from_list('custom_precip', colors, N=len(levels)-1)
        
        label = 'Total Precipitation (mm/day)'
        title_prefix = "Total Precipitation"
        
        vmin, vmax = None, None
        plot_kwargs = {'levels': levels, 'extend': 'neither'}

    if CARTOPY_AVAILABLE:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([32.5, 48.5, 3, 15], crs=ccrs.PlateCarree())
        
        # Enhanced Geography
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333')
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.5, edgecolor='#000000')
        ax.add_feature(cfeature.LAKES, edgecolor='#0000ff', facecolor='#add8e6', alpha=0.3)
        
        # Plot data with robust colorbar
        plot_params = {
            'ax': ax, 
            'transform': ccrs.PlateCarree(),
            'cmap': cmap, 
            'robust': True if 'tercile' not in var_name and 'anomaly' not in var_name else False,
            'vmin': vmin,
            'vmax': vmax,
            'add_colorbar': True,
            'cbar_kwargs': {
                'label': label,
                'shrink': 0.8,
                'pad': 0.08,
                'aspect': 25
            }
        }
        plot_params.update(plot_kwargs)
        
        im = ds[var_name].plot(**plot_params)
        
        # Fix colorbar ticks for terciles
        if 'tercile' in var_name:
            cb = im.colorbar
            cb.set_ticks([-1, 0, 1])
            cb.set_ticklabels(['Below Normal', 'Near Normal', 'Above Normal'])
        
        # Robust Gridlines and Labels
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4, color='black')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12, 'weight': 'bold', 'color': 'black'}
        gl.ylabel_style = {'size': 12, 'weight': 'bold', 'color': 'black'}
        
        # Re-asserting axis text to ensure visibility after tight_layout
        ax.text(-0.12, 0.5, 'LATITUDE', va='center', ha='center',
                rotation='vertical', transform=ax.transAxes, fontsize=14, fontweight='bold')
        ax.text(0.5, -0.12, 'LONGITUDE', va='center', ha='center',
                transform=ax.transAxes, fontsize=14, fontweight='bold')

    else:
        plot_params = {'cmap': cmap, 'robust': True, 'vmin': vmin, 'vmax': vmax}
        plot_params.update(plot_kwargs)
        ds[var_name].plot(**plot_params)
    
    date_str = str(lead_date)[:10] if not isinstance(lead_date, xr.DataArray) else str(lead_date.values)[:10]
    init_str = str(init_date)[:10] if not isinstance(init_date, xr.DataArray) else str(init_date.values)[:10]
    
    # Calculate target week string
    try:
        start_dt = datetime.strptime(date_str, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=6)
        target_week_str = f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d')}"
    except:
        target_week_str = date_str

    plt.title(f"ET-NeuralCast S2S Forecast: {title_prefix}\nInit: {init_str} | Target Week: {target_week_str} (Lead: {lead_weeks} Weeks)", 
              fontsize=18, pad=40, fontweight='bold', color='black')
    
    # Save with solid white background (the dashboard will invert labels to white in Dark Mode)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white', transparent=False)
    plt.close()


def run_inference(model_path, normalizer_path, pressure_file, sst_file=None, output_dir='outputs', lead_weeks=LEAD_WEEKS):
    """
    Run forecast inference using recent ERA5/ERA5T data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Normalizer
    normalizer = S2SNormalizer()
    normalizer.load(normalizer_path)
    
    # 2. Load Model
    model = UNetGenerator(input_channels=5, output_channels=1, target_size=(48, 60)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Load and Preprocess Latest ERA5 Data
    print(f"Loading pressure data: {pressure_file}")
    ds_pressure = xr.open_dataset(pressure_file)
    
    if sst_file:
        print(f"Loading SST data: {sst_file}")
        ds_sst = xr.open_dataset(sst_file)
        ds = xr.merge([ds_pressure, ds_sst])
    else:
        ds = ds_pressure
        print("Warning: No SST file provided.")

    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    
    recent_ds = ds.resample(time='1W').mean().isel(time=-1)
    forecast_date = recent_ds.time.values
    if isinstance(forecast_date, np.ndarray): forecast_date = forecast_date[0]
    
    lead_date_np = forecast_date + np.timedelta64(lead_weeks, 'W')
    lead_date = xr.DataArray(lead_date_np)
    lead_week = int(lead_date.dt.isocalendar().week)
    
    print(f"Generating forecast for week {lead_week} of {str(lead_date_np)[:4]} (Lead: {lead_weeks} weeks)")
    
    input_vars = ['z', 'q', 'u', 'v', 'sst']
    input_tensor = normalizer.transform_input(recent_ds, input_vars)
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    prediction_normalized = output_tensor.cpu().numpy()[0, 0]
    prediction_anomaly = prediction_normalized * normalizer.std['precip']
    climatology = normalizer.climatology['precip'][lead_week]
    prediction_total = np.maximum(prediction_anomaly + climatology, 0.0)
    
    # Percent of Normal
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_of_normal = (prediction_total / (climatology + 1e-8)) * 100
        percent_of_normal = np.nan_to_num(percent_of_normal, nan=100.0)
    
    # Tercile Category (-1: BN, 0: NN, 1: AN)
    tercile_cat = normalizer.get_tercile_category(prediction_total, lead_week, 'precip')
    
    lats = np.linspace(3, 15, 48)
    lons = np.linspace(33, 48, 60)
    
    pred_ds = xr.Dataset(
        data_vars={
            'precip_anomaly': (['lat', 'lon'], prediction_anomaly),
            'precip_total': (['lat', 'lon'], prediction_total),
            'precip_percent': (['lat', 'lon'], percent_of_normal),
            'precip_tercile': (['lat', 'lon'], tercile_cat)
        },
        coords={'lat': lats, 'lon': lons},
        attrs={'description': f'Forecast for {str(lead_date_np)[:10]}'}
    )
    
    pred_ds = mask_to_ethiopia(pred_ds)
    lead_suffix = f"_W{lead_weeks}"
    output_nc = os.path.join(output_dir, f"forecast_{str(lead_date_np)[:10]}{lead_suffix}.nc")
    pred_ds.to_netcdf(output_nc)
    
    # Visualizations
    visualize_forecast(pred_ds, 'precip_anomaly', forecast_date, lead_date_np, lead_weeks, os.path.join(output_dir, f"forecast_{str(lead_date_np)[:10]}{lead_suffix}_anomaly.png"))
    visualize_forecast(pred_ds, 'precip_total', forecast_date, lead_date_np, lead_weeks, os.path.join(output_dir, f"forecast_{str(lead_date_np)[:10]}{lead_suffix}_total.png"))
    visualize_forecast(pred_ds, 'precip_percent', forecast_date, lead_date_np, lead_weeks, os.path.join(output_dir, f"forecast_{str(lead_date_np)[:10]}{lead_suffix}_percent.png"))
    visualize_forecast(pred_ds, 'precip_tercile', forecast_date, lead_date_np, lead_weeks, os.path.join(output_dir, f"forecast_{str(lead_date_np)[:10]}{lead_suffix}_tercile.png"))

    print("Inference completed successfully.")


if __name__ == "__main__":
    MODEL_PATH = "checkpoints/G_warmstart_best.pth"
    NORMALIZER_PATH = "checkpoints/normalizer.pkl"
    PL_DIR, SST_DIR = "data/train/era5_pressure", "data/train/era5_sst"
    
    if os.path.exists(PL_DIR):
        pl_files = sorted([f for f in os.listdir(PL_DIR) if f.endswith('.nc')], reverse=True)
        if pl_files:
            pl_filename = pl_files[0]
            pl_file = os.path.join(PL_DIR, pl_filename)
            sst_file = os.path.join(SST_DIR, f"era5_sst_{pl_filename.replace('era5_pressure_', '')}")
            if not os.path.exists(sst_file): sst_file = None
            run_inference(MODEL_PATH, NORMALIZER_PATH, pl_file, sst_file)
        else: print("No ERA5 pressure files found.")
    else: print(f"Directory {PL_DIR} not found.")
