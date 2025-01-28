import xarray as xr
from pathlib import Path
import logging
import pdb

import functions.data_wrangling as data
import functions.eddy_feedback as ef

import warnings
warnings.filterwarnings("ignore", category=xr.SerializationWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

data_path = Path('/home/links/ct715/data_storage/PAMIP/monthly')
pd_path = data_path / '1.1_pdSST-pdSIC/combined_ua_epfy_divFy' 
fut_path = data_path / '1.6_pdSST-futArcSIC/combined_ua_epfy_divFy' 

list_dir = fut_path.iterdir()
model_list = [item.name for item in list_dir]
model_list.sort()
# model_list.remove('OpenIFS-511')
model_list.remove('OpenIFS-159')
model_list.remove('CESM1-WACCM-SC')
model_list.remove('EC-EARTH3')
model_list.remove('CNRM-CM6-1')
model_list.remove('IPSL-CM6A-LR')


for model in model_list:
    logging.info(f'Processing model: {model}')
    # import present day datasets and take winter mean
    pd_save_path = Path('/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC')
    pd_save_path.mkdir(exist_ok=True, parents=True)
    pd_output_file = pd_save_path / f'{model}_1.1_u_ubar_epfy_divFy.nc'
    if pd_output_file.exists():
        pass
    else:
        try:
            pd_model_path = pd_path / model
            pd_files = pd_model_path.glob('*.nc')
            
            # pdb.set_trace()
            
            ds_pd = xr.open_mfdataset(
                pd_files,
                #parallel=True,
                combine='nested',
                concat_dim='ens_ax',
                chunks={'time': 31},
                engine='netcdf4'
            )
        except KeyError as e:
            logging.info(f'Error with {model}: {e}')
            
        logging.info(f'Calculating pd ubar and saving file...')
        ds_pd['ubar'] = ds_pd.u.mean('lon').load()
        ds_pd.to_netcdf(pd_output_file)
    
    # import futArc datasets and take winter mean
    fut_save_path = Path('/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.6_pdSST-futArcSIC')
    fut_save_path.mkdir(exist_ok=True, parents=True)
    fut_output_file = fut_save_path / f'{model}_1.6_u_ubar_epfy_divFy.nc'
    if fut_output_file.exists():
        pass
    else:
        try:
            fut_model_path = fut_path / model
            fut_files = fut_model_path.glob('*.nc')
            ds_fut = xr.open_mfdataset(
                fut_files,
                #parallel=True,
                combine='nested',
                concat_dim='ens_ax',
                chunks={'time': 31},
                engine='netcdf4'
            )
        except KeyError as e:
            logging.info(f'Error with {model}: {e}')
            
        logging.info(f'Calculating fut ubar and saving file...')
        ds_fut['ubar'] = ds_fut.u.mean('lon').load()
        ds_fut.to_netcdf(fut_output_file)

