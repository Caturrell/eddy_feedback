import xarray as xr
import os

# Directories
base_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily'
sixh_dir = os.path.join(base_dir, 'k123_6h_QG_epfluxes')
daily_dir = os.path.join(base_dir, 'k123_daily_QG_epfluxes')
output_dir = os.path.join(base_dir, 'processed_efp')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def process_directory(directory, output_filename):
    """Process all NetCDF files in a directory, compute monthly averages, and save to output."""
    files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    files.sort()

    datasets = []
    for file in files:
        filepath = os.path.join(directory, file)
        print(f"Processing {file}...")
        ds = xr.open_dataset(filepath)
        # Resample to monthly, taking the mean
        ds_monthly = ds.resample(time='1MS').mean()
        datasets.append(ds_monthly)
        ds.close()

    # Concatenate all monthly datasets along the time dimension
    ds_all = xr.concat(datasets, dim='time')
    output_path = os.path.join(output_dir, output_filename)
    ds_all.to_netcdf(output_path)
    print(f"Monthly averages saved to {output_path}")

# Process 6-hourly data
print("Processing 6-hourly data...")
process_directory(sixh_dir, 'k123_6h_monthly_averages.nc')

# Process daily data
print("Processing daily data...")
process_directory(daily_dir, 'k123_daily_monthly_averages.nc')
