import xarray as xr
import numpy as np
import os
import glob

# Define paths
ua_path = '/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/6hrPlevPt/ua/gr/latest'
va_path = '/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/6hrPlevPt/va/gr/latest'
save_path = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical/IPSL-CM6A-LR/1850_2015/6hrPlevPt/yearly_data'

# Create save directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Get all ua files
ua_files = sorted(glob.glob(os.path.join(ua_path, '*.nc')))
print(f'Found {len(ua_files)} ua files to process')

# Process each file pair
for ua_file in ua_files:
    # Construct corresponding va file path
    va_file = ua_file.replace('ua', 'va')
    
    if not os.path.exists(va_file):
        print(f'Warning: va file not found for {ua_file}, skipping...')
        continue
    
    print(f'\nProcessing {os.path.basename(ua_file)} and {os.path.basename(va_file)}')
    
    try:
        # Open datasets
        ds_ua = xr.open_dataset(ua_file)
        ds_va = xr.open_dataset(va_file)
        
        # Combine datasets
        ds = xr.merge([ds_ua, ds_va])
        
        # Get the years present in the dataset
        years = np.unique(ds['time'].dt.year.values)
        print(f'  Years in file: {years[0]} to {years[-1]}')
        
        # Process each year
        for year in years:
            # Select just this year's data
            ds_year = ds.sel(time=ds['time'].dt.year == year)
            
            # Construct descriptive output filename
            output_file = os.path.join(
                save_path,
                f'{year}_IPSL-CM6A-LR_historical_r1i1p1f1_uv.nc'
            )
            
            # Save the yearly chunk
            ds_year.to_netcdf(output_file, compute=True)
            print(f'  Saved {year} ({len(ds_year.time)} timesteps) -> {os.path.basename(output_file)}')
        
        # Close datasets to free memory
        ds.close()
        ds_ua.close()
        ds_va.close()
        
    except Exception as e:
        print(f'Error processing files: {e}')
        continue

print('\nProcessing complete!')