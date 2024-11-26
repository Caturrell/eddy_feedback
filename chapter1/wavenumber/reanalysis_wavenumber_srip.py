"""
    A script to calculate various EFP with different wavenumbers.
    Wavenumbers 1,2,3 have been calculated by SRIP.
    
    !!!!INCOMPLETE!!!!
    
"""

import xarray as xr
from pathlib import Path
import logging

import functions.eddy_feedback as ef

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#--------------------------------------------------------------------------------------------------

# set paths and models
data_path = Path('/home/links/ct715/data_storage/reanalysis/srip_datasets')
models = ['JRA55', 'NCEP-NCAR', 'ERA-Interim']

logging.info('Importing datasets.')
ds = {}
for model in models:
    
    # set paths and import datasets into one set
    dataset_paths = data_path.glob(f'{model}_*_monthly_original.nc')
    dataset = xr.open_mfdataset(
        dataset_paths,
        parallel=True
    )
    
    # bespoke changes
    dataset = dataset.rename({'u': 'ubar'})

    # save to dataset dict
    ds[model] = dataset
    
#--------------------------------------------------------------------------------------------------

variables_list = ['EPFD_phi_pr', 'EPFD_phi_pr_k1', 'EPFD_phi_pr_k2', 'EPFD_phi_pr_k3']

for var in variables_list:
    for model in ds:
        
        efp_nh = ef.calculate_efp(model, data_type='reanalysis', which_div1=var)
        efp_sh = ef.calculate_efp(model, data_type='reanalysis', which_div1=var, calc_south_hemis=True)
        
        print(f'{model}: EFP value for {var}')
