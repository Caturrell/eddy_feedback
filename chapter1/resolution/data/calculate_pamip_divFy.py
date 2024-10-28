import xarray as xr

import pdb
import warnings
import logging
import os

import functions.eddy_feedback as ef
import functions.data_wrangling as data

# define function
def process_monthly_pamip_data(ua, epfy):
    """ Process ua and epfy variables, put into new dataset,
        calculate ubar and divFy """
    model = xr.Dataset({'ubar': ua.u.mean('lon'), 'epfy': epfy.epfy})
    model = ef.calculate_divFphi(model)
    return model


# remove_models = ['FGOALS-f3-L', 'IPSL-CM6A-LR', 'MIROC6_mass', 'HadGEM3-GC31-LL', \
#                  'OpenIFS-1279', 'OpenIFS-511', 'OpenIFS-159']



if __name__ == '__main__':
    
    # disable SerializationWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=xr.SerializationWarning)
    
    # Configure logging to write to a file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
)
    
    # set directory path
    path = '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/'
    # # Extract model names from the directory
    # model_names = os.listdir(path + '/ua')
    # model_names.sort()
    
    missing_data_models = ['CanESM5', 'CESM2', 'HadGEM3-GC31-MM', 'MIROC6', 'NorESM2-LM']
    model_names = missing_data_models

    # Loop over each model
    for model in model_names:
        logging.info(f'Processing {model}...')
        
        # define paths for data of each variable
        ua_data_path = os.path.join(path, 'ua', model, '*.nc')
        epfy_data_path = os.path.join(path, 'epfy', model, '*.nc')
        
        # Open ua dataset
        try:
            ua = xr.open_mfdataset(
                ua_data_path, 
                cache=False, 
                parallel=True, 
                chunks={'time': 31},
                concat_dim='ens_ax', 
                combine='nested'
            )
            ua = data.data_checker1000(ua)
        except Exception as e:
            logging.error(f'Error opening dataset for {model}, variable ua: {e}')
            continue  # Skip to the next variable/model if there's an error
        # Open epfy dataset
        try:
            epfy = xr.open_mfdataset(
                epfy_data_path, 
                cache=False, 
                parallel=True, 
                chunks={'time': 31},
                concat_dim='ens_ax', 
                combine='nested'
            )
            epfy = data.data_checker1000(epfy)
        except Exception as e:
            logging.error(f'Error opening dataset for {model}, variable epfy: {e}')
            continue  # Skip to the next variable/model if there's an error
        
        if not ua.level.equals(epfy.level):
            if len(ua.level) > len(epfy.level):
                ua = ua.interp(level=epfy.level)
                epfy = epfy
            else:
                ua = ua
                epfy = epfy.interp(level=ua.level)
        else:
            ua, epfy = ua, epfy  # Use original DataArrays if levels match

        
        # now attempt to process data
        try:
            ds = process_monthly_pamip_data(ua, epfy)
        except Exception as e:
            logging.error(f'Error processing {model}.')
            logging.error(f'{e}')
            continue
        
        ds.to_netcdf(f'/home/links/ct715/data_storage/PAMIP/processed_monthly/{model}_1.1_pdSST-pdSIC_ubar_epfy_divFy.nc')
        logging.info(f'Processed dataset for {model} saved.')
        