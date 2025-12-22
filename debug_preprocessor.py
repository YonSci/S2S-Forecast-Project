"""
Debug preprocessor to find where lat dimension collapses
"""
import xarray as xr
import os

# Load CHIRPS
chirps_ds = xr.open_dataset("data/raw/chirps/chirps-v2.0.2000.days_p25_clip.nc")
print(f"Original CHIRPS shape: {chirps_ds['precip'].shape}")
print(f"Original CHIRPS dims: {chirps_ds['precip'].dims}")

# Resample to weekly
chirps_weekly = chirps_ds.resample(time='W-SUN').mean()
print(f"\nWeekly CHIRPS shape: {chirps_weekly['precip'].shape}")
print(f"Weekly CHIRPS dims: {chirps_weekly['precip'].dims}")

# Try selecting a specific time
common_time = chirps_weekly.time.values[0]
print(f"\nSelecting time: {common_time}")
selected = chirps_weekly.sel(time=common_time)
print(f"Selected shape: {selected['precip'].shape}")
print(f"Selected dims: {selected['precip'].dims}")
print(f"Selected precip values shape: {selected['precip'].values.shape}")

# Try isel instead
print(f"\n=== Using isel ===")
selected_isel = chirps_weekly.isel(time=0)
print(f"isel shape: {selected_isel['precip'].shape}")
print(f"isel dims: {selected_isel['precip'].dims}")
