"""
Quick debugging script to inspect CHIRPS and ERA5 data structures
"""
import xarray as xr

print("=== CHIRPS Inspection ===")
chirps_ds = xr.open_dataset("data/raw/chirps/chirps-v2.0.2000.days_p25_clip.nc")
print(f"Dims: {chirps_ds.dims}")
print(f"Coords: {list(chirps_ds.coords)}")
print(f"Data vars: {list(chirps_ds.data_vars)}")
print(f"Lat range: {chirps_ds.lat.min().values} to {chirps_ds.lat.max().values}")
print(f"Lon range: {chirps_ds.lon.min().values} to {chirps_ds.lon.max().values}")

# Test weekly resample
print("\n=== CHIRPS Weekly Resample Test ===")
chirps_weekly = chirps_ds.resample(time='W-SUN').mean()
print(f"Weekly dims: {chirps_weekly.dims}")
print(f"Weekly coords: {list(chirps_weekly.coords)}")
print(f"Weekly lat range: {chirps_weekly.lat.min().values} to {chirps_weekly.lat.max().values}")

# Test precip extraction
print("\n=== Precip Variable ===")
prec = chirps_weekly['precip'].isel(time=0)
print(f"Precip shape: {prec.shape}")
print(f"Precip dims: {prec.dims}")
print(f"Precip values shape: {prec.values.shape}")

print("\n=== ERA5 Inspection ===")
era5_ds = xr.open_dataset("data/raw/era5_pressure/era5_pressure_2000_01.nc")
print(f"Dims: {era5_ds.dims}")
print(f"Coords: {list(era5_ds.coords)}")
print(f"Data vars: {list(era5_ds.data_vars)}")
