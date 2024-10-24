import xarray as xr
import os
import logging

# Configure logging to write to a file
log_file = '/home/links/ct715/eddy_feedback/chapter2/resolution/model_units_check.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Extract model names from the directory
path = '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/epfy/'
model_names = os.listdir(path)
model_names.sort()

# Loop over each model to check units for the variables and ensure no corrupt files
for model in model_names:
    logging.info(f'Checking {model}')
    
    for variable in ['ua', 'epfy']:
        # Define path to both sets of data for each model
        var_path = f'/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/{variable}/'
        model_data_path = os.path.join(var_path, model, '*.nc')
        
        # Open dataset
        try:
            ds = xr.open_mfdataset(
                model_data_path, 
                cache=False, 
                parallel=True, 
                chunks={'time': 31},
                concat_dim='ens_ax', 
                combine='nested'
            )
        except Exception as e:
            logging.error(f'Error opening dataset for {model}, variable {variable}: {e}')
            continue  # Skip to the next variable/model if there's an error
        
        # Check units
        try:
            units = ds[variable].attrs.get('units', 'No units specified')
        except KeyError:
            units = 'Variable not found'
            logging.error(f'ISSUE: {model} - {variable} not found in dataset')
        
        logging.info(f'{model}: {variable}: {units}')
