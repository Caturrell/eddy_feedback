""" This script is for data management on JASMIN only (due to seemingly corrupt files)"""

import xarray as xr
from pathlib import Path
import logging
import pdb

import functions.data_wrangling as data

import warnings
warnings.filterwarnings("ignore", category=xr.SerializationWarning)

# set data paths and create model list
raw_path = Path('/home/links/ct715/data_storage/PAMIP/monthly')
exp_names = ['1.1_pdSST-pdSIC','1.6_pdSST-futArcSIC']

pd_ua, fut_ua = {}, {}
for model in model_list:
    for exp in exp_names:
        
        raw_data_path = raw_path / f'{exp}/ua/{model}' 
        
        # First, import data (some aren't in this location)
        if model in ['CESM1-WACCM-SC', 'IPSL-CM6A-LR']:
            ds = xr.open_mfdataset(
                str(raw_data_path / '*.nc'),
                parallel=True
            )
        else:
            ds = xr.open_mfdataset(
                str(raw_data_path / '*.nc'),
                parallel=True,
                combine='nested',
                concat_dim='ens_ax',
                engine='netcdf4'
            )
            
        # Second, extract ua (some are U)
        if 'ua' in ds.data_vars:
            print(f'{model} ({exp}) has ua')
        else:
            print(f'{model} ({exp}) has: {ds.data_vars}')
            
        if exp == exp_names[0]:
            pd_ua[model] = ds
        else:
            fut_ua[model] = ds



pdb.set_trace()