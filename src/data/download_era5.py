import cdsapi
import os

def download_era5_pressure(c, year, month, output_dir):
    """
    Downloads ERA5 pressure level data (Wind, Geopotential, Humidity).
    Domain: 40S - 40N, 0E - 100E (Indian Ocean + Africa + West Pacific)
    """
    filename = f"era5_pressure_{year}_{month}.nc"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        print(f"Skipping {filename} (Exists)")
        return

    print(f"Downloading {filename}...")
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'specific_humidity', 'u_component_of_wind',
                'v_component_of_wind',
            ],
            'pressure_level': [
                '200', '500', '850',
            ],
            'year': year,
            'month': month,
            'day': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
                '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24',
                '25', '26', '27', '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00', '18:00',
            ],
            'area': [
                40, 0, -40, 100, # North, West, South, East
            ],
            'grid': [1.0, 1.0], # Coarser grid to save space (1 degree)
        },
        output_path)

def download_era5_sst(c, year, month, output_dir):
    """
    Downloads ERA5 Sea Surface Temperature.
    Domain: Global or larger domain for teleconnections.
    """
    filename = f"era5_sst_{year}_{month}.nc"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"Skipping {filename} (Exists)")
        return

    print(f"Downloading {filename}...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'sea_surface_temperature',
            'year': year,
            'month': month,
            'day': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
                '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24',
                '25', '26', '27', '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00', '18:00',
            ],
            'grid': [1.0, 1.0],
            # SST needs global coverage for ENSO (Pacific) and IOD (Indian)
            'area': [60, -180, -60, 180], 
        },
        output_path)

if __name__ == "__main__":
    try:
        c = cdsapi.Client()
    except Exception as e:
        print("Error connecting to CDS API. Make sure you have a .cdsapirc file.")
        print(e)
        exit(1)

    START_YEAR = 2006
    END_YEAR = 2009
    OUTPUT_DIR_PL = os.path.join(os.getcwd(), "data", "raw", "era5_pressure")
    OUTPUT_DIR_SST = os.path.join(os.getcwd(), "data", "raw", "era5_sst")
    
    os.makedirs(OUTPUT_DIR_PL, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SST, exist_ok=True)

    months = [str(i).zfill(2) for i in range(1, 13)]
    
    for year in range(START_YEAR, END_YEAR + 1):
        for month in months:
            download_era5_pressure(c, str(year), month, OUTPUT_DIR_PL)
            download_era5_sst(c, str(year), month, OUTPUT_DIR_SST)
