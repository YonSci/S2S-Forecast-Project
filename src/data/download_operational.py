import cdsapi
import os
from datetime import datetime, timedelta

def download_operational_data(output_dir_pl, output_dir_sst):
    """
    Downloads the most recent available ERA5/ERA5T data for inference.
    ERA5T (Preliminary) usually has a 5-day delay.
    """
    # Initialize CDS client - will prefer environment variables if available
    # (CDSAPI_URL, CDSAPI_KEY)
    c = cdsapi.Client(
        url=os.environ.get('CDSAPI_URL'), 
        key=os.environ.get('CDSAPI_KEY')
    )
    
    # Get current date and subtract 5 days to ensure data is available (ERA5T delay)

    target_date = datetime.now() - timedelta(days=5)
    year = str(target_date.year)
    month = str(target_date.month).zfill(2)
    # Get all days available in that month up to target_date
    days = [str(d).zfill(2) for d in range(1, target_date.day + 1)]
    
    print(f"Targeting Operational Data for: {year}-{month}")
    
    # 1. Download Pressure Levels
    pl_filename = f"era5_pressure_{year}_{month}.nc"
    pl_path = os.path.join(output_dir_pl, pl_filename)
    
    print(f"Downloading Pressure Data to: {pl_path}")
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'specific_humidity', 'u_component_of_wind',
                'v_component_of_wind',
            ],
            'pressure_level': ['200', '500', '850'],
            'year': year,
            'month': month,
            'day': days,
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [40, 0, -40, 100],
            'grid': [1.0, 1.0],
        },
        pl_path)

    # 2. Download SST
    sst_filename = f"era5_sst_{year}_{month}.nc"
    sst_path = os.path.join(output_dir_sst, sst_filename)
    
    print(f"Downloading SST Data to: {sst_path}")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'sea_surface_temperature',
            'year': year,
            'month': month,
            'day': days,
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [60, -180, -60, 180],
            'grid': [1.0, 1.0],
        },
        sst_path)

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.getcwd(), "data", "raw")
    PL_DIR = os.path.join(DATA_DIR, "era5_pressure")
    SST_DIR = os.path.join(DATA_DIR, "era5_sst")
    
    os.makedirs(PL_DIR, exist_ok=True)
    os.makedirs(SST_DIR, exist_ok=True)
    
    try:
        download_operational_data(PL_DIR, SST_DIR)
        print("\n[+] Operational data download complete!")
    except Exception as e:
        print(f"\n[-] Error downloading operational data: {e}")
        print("Note: If the month has just started, you might need to request the previous month's data.")
