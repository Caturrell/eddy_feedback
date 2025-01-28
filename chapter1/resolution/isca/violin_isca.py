import xarray as xr
from pathlib import Path
import seaborn as sns 
import matplotlib.pyplot as plt
import logging

import splitting_isca_data as sid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

isca_path = Path('/home/links/ct715/data_storage/isca/held-suarez')
isca_exp_list = isca_path.iterdir()

exp_list = sorted(item.name for item in isca_exp_list)
exp_list = [item for item in exp_list if item.startswith('HS')]

datasets = {}
for exp in exp_list:
    
    nc_path = Path(isca_path) / exp
    # nc_files = list(nc_path.glob('*.nc'))
    nc_files = nc_path.glob('*.nc')
    
    ds = xr.open_mfdataset(
        nc_files,
        parallel=True,
        chunks={'time': 360}
    )
    
    # datasets[exp] = ds.load()
    ds_seasonal = sid.seasonal_mean_datasets(ds)
    datasets[exp] = ds_seasonal.load()