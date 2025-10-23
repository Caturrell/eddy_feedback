import logging
import os
import xarray as xr


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

source_path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/'
time_freq_folder = ['6h']#, 'daily']
different_methods_folder = ['pr', 'QG']

dest_path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/various_efp_methods'

for freq in time_freq_folder:
    
    # set final file name
    save_file_name = f'{freq}_ubar_epf-pr-QG_1MS_1958-2016.nc'
    save_path = os.path.join(dest_path, save_file_name)
    if os.path.exists(save_path):
        logging.info(f"File already exists: {save_path}. Skipping...")
        continue
    
    dataset = {}
    for method in different_methods_folder:
        
        # set path location where data is stored in split-yearly netCDF files
        if freq == 'daily':
            start_path = os.path.join(source_path, f'{freq}_uvtw', f'{method}_epf_uvtw')
        else:
            start_path = os.path.join(source_path, f'{freq}_uvtw', 'daily_averages', f'{method}_epf_uvtw')
        
        dataset[method] = xr.open_mfdataset(f'{start_path}/*.nc', combine='by_coords')
        dataset[method] = dataset[method][['ubar', f'div1_{method}']]
        dataset[method] = dataset[method].resample(time='1MS').mean()     
        logging.info(f"Completed wrangling for: {method} at {freq} frequency")
        
    ds = xr.merge([dataset['pr'], dataset['QG']])
    ds = ds[['ubar', 'div1_pr', 'div1_QG']]
    logging.info(f"Saving wrangled data to: {save_path}")
    ds.to_netcdf(save_path)
    
    logging.info('Processing complete.')
    
