"""
Debug CHIRPS cropping to Ethiopia
"""
import xarray as xr

# Load original CHIRPS
chirps_ds = xr.open_dataset("data/raw/chirps/chirps-v2.0.2000.days_p25_clip.nc")
print(f"Original CHIRPS:")
print(f"  Dims: {chirps_ds.dims}")
print(f"  Lat range: {chirps_ds.lat.min().values} to {chirps_ds.lat.max().values}")
print(f"  Lon range: {chirps_ds.lon.min().values} to {chirps_ds.lon.max().values}")

# Test the Ethiopia bbox slicing
ETHIOPIA_BBOX = {
    'lat_min': 3,
    'lat_max': 15,
    'lon_min': 33,
    'lon_max': 48
}

print(f"\nTarget Ethiopia bbox:")
print(f"  Lat: {ETHIOPIA_BBOX['lat_min']} to {ETHIOPIA_BBOX['lat_max']}")
print(f"  Lon: {ETHIOPIA_BBOX['lon_min']} to {ETHIOPIA_BBOX['lon_max']}")

# Try slicing
print(f"\nTrying slice...")
cropped = chirps_ds.sel(
    lat=slice(ETHIOPIA_BBOX['lat_max'], ETHIOPIA_BBOX['lat_min']),
    lon=slice(ETHIOPIA_BBOX['lon_min'], ETHIOPIA_BBOX['lon_max'])
)

print(f"Cropped dims: {cropped.dims}")
print(f"Cropped lat range: {cropped.lat.min().values if len(cropped.lat) > 0 else 'EMPTY'}")
print(f"Cropped precip shape: {cropped['precip'].shape}")

# The file is already clipped! Let's check if it needs cropping at all
print(f"\n=== File is already clipped to Ethiopia! ===")
print("The filename contains 'clip' - it's pre-cropped. No bbox slicing needed!")
