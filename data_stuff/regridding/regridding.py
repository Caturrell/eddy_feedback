"""Python function for regridding data."""

import sys
import glob
import os

import numpy as np
import xarray as xr
import xesmf as xe

import functions.data_wrangling as data
sys.path.append('/home/users/cturrell/documents/eddy_feedback')

#==================================================================================================

#----------------------
# REGRIDDING FUNCTIONS
#----------------------

# regrid PAMIP data
def regrid_dataset_3x3(dataset, check_dims=False):
    """
    Input: Xarray dataset
            - Must contain (lat, lon)
            
    Output: Regridded Xarray dataset 
            to 3 deg lat lon
    
    """
    # Rename
    if check_dims:
        dataset = data.check_dimensions(dataset)
    # build regridder
    ds_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90, 93, 3)),
            'lon': (['lon'], np.arange(0,360, 3))
        }
    )

    regridder = xe.Regridder(dataset, ds_out, "bilinear")
    ds_new = regridder(dataset)

    # verify that the result is the same as regridding each variable one-by-one
    for k in dataset.data_vars:
        print(k, ds_new[k].equals(regridder(dataset[k])))

    print('Regridding and checks complete. Dataset ready.')
    return ds_new

#=================================================================================================

if __name__ == '__main__':

    # Set var list
    variables = ['ta', 'ua', 'va']

    # Set time of day for program echo
    from datetime import datetime
    now = datetime.now().strftime("%H:%M:%S")
    print("Current Time =", now)

    for var in variables:

        # reset time for counter
        now = datetime.now().strftime("%H:%M:%S")

        print(f'[{now}]: Importing paths for CNRM-CM6-1 {var}...')
        files = glob.glob(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/daily/ \
                          {var}/pdSST-pdSIC/CNRM-CM6-1/*.nc')
        for count, item in enumerate(files):

            # reset current time
            now = datetime.now().strftime("%H:%M:%S")

            print(f'[{now}]: Opening ensemble member {var}: {count+1}')
            ds = xr.open_mfdataset(item)
            ds = ds[[f'{var}']]

            print('Starting regridding function...')
            ds = regrid_dataset_3x3(ds, check_dims=True)

            print('Saving dataset...')
            ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/ \
                         CNRM-CM6-1_3x3/{var}/{os.path.basename(item)}')

        print(f'Variable {var} finished.')

    print('PROGRAM COMPLETE.')
