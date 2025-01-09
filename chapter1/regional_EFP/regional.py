import xarray as xr
import jsmetrics as js
from pathlib import Path
import logging

import functions.data_wrangling as data

import warnings
warnings.filterwarnings("ignore", category=xr.SerializationWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_ua_models(model_list):
    
    for model in model_list:
        logging.info(f'Processing: {model}')
        
        try:
            pd_dataset = xr.open_mfdataset(
                f'/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/ua/{model}/*.nc',
                parallel=True,
                combine='nested',
                concat_dim='ens_ax',
                engine='netcdf4'
            )
            pd_dataset.load()
            pd_dataset = data.data_checker1000(pd_dataset)
            
            fut_dataset = xr.open_mfdataset(
                f'/home/links/ct715/data_storage/PAMIP/monthly/1.6_pdSST-futArcSIC/ua/{model}/*.nc',
                parallel=True,
                combine='nested',
                concat_dim='ens_ax',
                engine='netcdf4'
            )
            fut_dataset.load()
            fut_dataset = data.data_checker1000(fut_dataset)
            
        except Exception as e:
            logging.debug(f'Exception raised: {e}')
        
        # save datasets in one file
        logging.info('Saving datasets...')
        save_path = Path('/home/links/ct715/data_storage/PAMIP/processed_monthly/ua-nc')
        pd_save = save_path / '1.1_pdSST-pdSIC' / f'{model}_1.1_ua.nc'
        fut_save = save_path / '1.6_pdSST-futArcSIC' / f'{model}_1.6_ua.nc'
        pd_dataset.to_netcdf(pd_save)
        fut_dataset.to_netcdf(fut_save)
        
if __name__ == '__main__':
    
    logging.info('Starting program.')
    data_path = Path('/home/links/ct715/data_storage/PAMIP/monthly')
    pd_path = data_path / '1.1_pdSST-pdSIC' / 'ua'
    fut_path = data_path / '1.6_pdSST-futArcSIC' / 'ua'
    # obtain list of models
    list_dir = list(fut_path.iterdir())
    model_list = [item.name for item in list_dir]
    model_list.sort()
    model_list.remove('CESM2')
    model_list.remove('EC-EARTH3')
    
    process_ua_models(model_list)
    
    
        
