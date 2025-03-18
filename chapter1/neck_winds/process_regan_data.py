"""
    Calculate EP fluxes for each of Regan's gamma=1,..,6 experiments,
    saving a dataset with monthly averages for all variables
"""

import xarray as xr
from pathlib import Path
import functions.eddy_feedback as ef
import functions.aos_functions as aos

# Define input and output paths
path = Path('/disco/share/rm811/processed')
save_location = Path('/home/links/ct715/data_storage/isca/regan_sims/polar_vortex')

# List of variables to load
var_list = ['T', 'u', 'v', 'w']

for i in range(1, 7):
    print(f'Processing gamma = {i}')
    exp_name = f'PK_e0v{i}z13_q6m2y45l800u200'

    # Generate save filename
    extract_exp = "_".join(exp_name.split('_')[:2])
    save_file_name = save_location / f'{extract_exp}_mon_uvt_ep.nc'

    # Check if file already exists
    if save_file_name.exists():
        print(f"File {save_file_name} already exists. Skipping...")
        continue  # Skip to the next iteration

    print('Importing data into dictionary...')
    
    # Dictionary for storing datasets
    ds = {}
    
    for var in var_list:
        data_path = path / f'{exp_name}_{var}.nc'
        print(f'Loading variable: {var}')
        
        ds[var] = xr.open_mfdataset(
            str(data_path), chunks={'time': 30}
        )  # No `.load()` to keep lazy loading

    print('Data loaded. Computing EP fluxes...')
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(
        ds['u'].ucomp, ds['v'].vcomp, ds['T'].temp, do_ubar=True
    )

    print('EP fluxes calculated. Computing monthly mean data...')
    
    # Perform resampling efficiently
    monthly_means = {
        'u': ds['u'].ucomp.resample(time="1MS").mean().assign_attrs(ds['u'].ucomp.attrs),
        'v': ds['v'].vcomp.resample(time="1MS").mean().assign_attrs(ds['v'].vcomp.attrs),
        't': ds['T'].temp.resample(time="1MS").mean().assign_attrs(ds['T'].temp.attrs),
        'epfy': ep1.resample(time="1MS").mean().assign_attrs(ep1.attrs),
        'epfz': ep2.resample(time="1MS").mean().assign_attrs(ep2.attrs),
        'divFy': div1.resample(time="1MS").mean().assign_attrs(div1.attrs),
        'divFz': div2.resample(time="1MS").mean().assign_attrs(div2.attrs),
    }

    print('Monthly averages calculated. Creating new Dataset...')
    
    new_ds = xr.Dataset(monthly_means)
    new_ds = new_ds.rename({'pfull': 'level'})

    # Optimize NetCDF saving (compression)
    comp = {'zlib': True, 'complevel': 4, 'shuffle': True}  # Balanced compression
    encoding = {var: comp for var in new_ds.data_vars}

    print(f'Saving dataset to {save_file_name}')
    new_ds.to_netcdf(save_file_name, encoding=encoding)

    print(f'Completed gamma = {i}.')

print('Program complete.')
