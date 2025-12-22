import xarray as xr
import numpy as np
import os
from glob import glob
from datetime import datetime, timedelta

def load_and_preprocess_chirps(chirps_dir, year, ethiopia_bbox=None):
    """
    Load CHIRPS precipitation data for a given year.
    
    Args:
        chirps_dir (str): Path to CHIRPS directory
        year (int): Year to load
        ethiopia_bbox (dict): Bounding box {'lat_min': 3, 'lat_max': 15, 'lon_min': 33, 'lon_max': 48}
    
    Returns:
        xr.Dataset: Daily precipitation data
    """
    # Find the file
    pattern = os.path.join(chirps_dir, f"chirps-v2.0.{year}.days*.nc")
    files = glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No CHIRPS file found for year {year}")
    
    print(f"Loading CHIRPS: {files[0]}")
    ds = xr.open_dataset(files[0])
    
    # Crop to Ethiopia if bbox provided AND file is not already clipped
    # Note: Files with 'clip' in the name are pre-cropped
    if ethiopia_bbox and '_clip' not in files[0]:
        ds = ds.sel(
            lat=slice(ethiopia_bbox['lat_min'], ethiopia_bbox['lat_max']),  # Ascending order
            lon=slice(ethiopia_bbox['lon_min'], ethiopia_bbox['lon_max'])
        )
    else:
        print("  File is already clipped, skipping bbox cropping")
    
    return ds

def load_and_preprocess_era5(era5_pressure_dir, era5_sst_dir, year, month):
    """
    Load ERA5 pressure and SST data for a given year-month.
    
    Returns:
        xr.Dataset: Combined ERA5 data
    """
    # Load pressure level data
    pressure_file = os.path.join(era5_pressure_dir, f"era5_pressure_{year}_{month:02d}.nc")
    sst_file = os.path.join(era5_sst_dir, f"era5_sst_{year}_{month:02d}.nc")
    
    if not os.path.exists(pressure_file):
        raise FileNotFoundError(f"ERA5 pressure file not found: {pressure_file}")
    if not os.path.exists(sst_file):
        raise FileNotFoundError(f"ERA5 SST file not found: {sst_file}")
    
    print(f"Loading ERA5: {year}-{month:02d}")
    
    # Load datasets
    ds_pressure = xr.open_dataset(pressure_file)
    ds_sst = xr.open_dataset(sst_file)
    
    # Aggregate from 6-hourly to daily
    ds_pressure_daily = ds_pressure.resample(valid_time='1D').mean()
    ds_sst_daily = ds_sst.resample(valid_time='1D').mean()
    
    # Merge
    ds_era5 = xr.merge([ds_pressure_daily, ds_sst_daily])
    
    return ds_era5

def create_weekly_samples(chirps_ds, era5_datasets, lead_weeks=1):
    """
    Create weekly samples with temporal lag.
    
    Args:
        chirps_ds: Daily CHIRPS data
        era5_datasets: List of daily ERA5 datasets
        lead_weeks: Forecast lead time in weeks
    
    Returns:
        list: [(era5_week_t, chirps_week_t+lead), ...]
    """
    samples = []
    
    # Concatenate all ERA5 data
    era5_combined = xr.concat(era5_datasets, dim='valid_time')
    era5_combined = era5_combined.rename({'valid_time': 'time'})
    
    # Make sure both use the same time coordinate name
    # Resample to weekly (Sunday anchor)
    era5_weekly = era5_combined.resample(time='W-SUN').mean()
    chirps_weekly = chirps_ds.resample(time='W-SUN').mean()
    
    # Use index-based selection instead of coordinate-based to avoid datetime precision issues
    num_weeks_era5 = len(era5_weekly.time)
    num_weeks_chirps = len(chirps_weekly.time)
    
    # Use the minimum number of weeks available
    num_weeks = min(num_weeks_era5, num_weeks_chirps)
    
    if num_weeks == 0:
        print("Warning: No weekly data available!")
        return samples
    
    # Create samples with lag
    for i in range(num_weeks - lead_weeks):
        input_week = era5_weekly.isel(time=i)
        target_week = chirps_weekly.isel(time=i + lead_weeks)
        
        samples.append((input_week, target_week))
    
    return samples

def preprocess_for_training(data_dir, years, ethiopia_bbox, lead_weeks=1):
    """
    Main preprocessing pipeline.
    
    Args:
        data_dir (str): Path to data/raw
        years (list): List of years to process
        ethiopia_bbox (dict): Ethiopia bounding box
        lead_weeks (int): Forecast lead time in weeks
    
    Returns:
        list: Training samples
    """
    chirps_dir = os.path.join(data_dir, 'chirps')
    era5_pressure_dir = os.path.join(data_dir, 'era5_pressure')
    era5_sst_dir = os.path.join(data_dir, 'era5_sst')
    
    all_samples = []
    
    for year in years:
        print(f"\n=== Processing Year {year} ===")
        
        # Load CHIRPS for the year
        chirps_ds = load_and_preprocess_chirps(chirps_dir, year, ethiopia_bbox)
        
        # Load ERA5 for all months
        era5_datasets = []
        for month in range(1, 13):
            try:
                ds = load_and_preprocess_era5(era5_pressure_dir, era5_sst_dir, year, month)
                era5_datasets.append(ds)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        # Create weekly samples
        samples = create_weekly_samples(chirps_ds, era5_datasets, lead_weeks=lead_weeks)
        all_samples.extend(samples)
        
        print(f"Created {len(samples)} samples for {year} (Lead: {lead_weeks} weeks)")
    
    return all_samples


if __name__ == "__main__":
    # Test the preprocessing
    DATA_DIR = "data/raw"
    YEARS = [2000, 2001]
    ETHIOPIA_BBOX = {
        'lat_min': 3,
        'lat_max': 15,
        'lon_min': 33,
        'lon_max': 48
    }
    
    samples = preprocess_for_training(DATA_DIR, YEARS, ETHIOPIA_BBOX)
    print(f"\nTotal samples created: {len(samples)}")
    
    if len(samples) > 0:
        input_sample, target_sample = samples[0]
        print(f"\nSample 0:")
        print(f"Input keys: {list(input_sample.data_vars)}")
        print(f"Target keys: {list(target_sample.data_vars)}")
