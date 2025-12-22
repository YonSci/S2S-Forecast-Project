
import xarray as xr
import sys

file_path = r"data\raw\chirps\chirps-v2.0.2022.days_p05.nc"

print(f"Checking file: {file_path}")

# Check magic bytes
try:
    with open(file_path, 'rb') as f:
        header = f.read(4)
        print(f"First 4 bytes: {header}")
        print(f"Hex: {header.hex()}")
        if header.startswith(b'CDF'):
            print("Magic bytes indicate NetCDF Classic/64-bit.")
        elif header.startswith(b'\x89HDF'):
            print("Magic bytes indicate HDF5 (NetCDF-4).")
        else:
            print("Magic bytes do NOT match standard NetCDF signatures.")
except Exception as e:
    print(f"Error reading header: {e}")

# Try explicit engine
print("\nAttempting to open with engine='netcdf4'...")
try:
    ds = xr.open_dataset(file_path, engine='netcdf4')
    print("Success with netcdf4!")
    ds.close()
except Exception as e:
    print(f"Failed with netcdf4: {e}")

# Try h5netcdf engine
print("\nAttempting to open with engine='h5netcdf'...")
try:
    ds = xr.open_dataset(file_path, engine='h5netcdf')
    print("Success with h5netcdf!")
    ds.close()
except Exception as e:
    print(f"Failed with h5netcdf: {e}")
