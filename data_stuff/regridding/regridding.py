"""Python function for regridding data."""

import sys
import glob
import os

import numpy as np
import xarray as xr
import xesmf as xe

sys.path.append('/home/users/cturrell/documents/eddy_feedback')
# pylint: disable=wrong-import-position
import functions.data_wrangling as data

#==================================================================================================

#----------------------
# REGRIDDING FUNCTIONS
#----------------------

# regrid PAMIP data
def regrid_dataset_3x3(ds, check_dims=False):
    """
    Input: Xarray dataset
            - Must contain (lat, lon)
            
    Output: Regridded Xarray dataset 
            to 3 deg lat lon
    
    """
    # Rename
    if check_dims:
        ds = data.check_dimensions(ds)
    # build regridder
    ds_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90, 93, 3)),
            'lon': (['lon'], np.arange(0,360, 3))
        }
    )

    regridder = xe.Regridder(ds, ds_out, "bilinear")
    ds_new = regridder(ds)

    # verify that the result is the same as regridding each variable one-by-one
    for k in ds.data_vars:
        print(k, ds_new[k].equals(regridder(ds[k])))

    print('Regridding and checks complete. Dataset ready.')
    return ds_new


# Regrid reanalysis data
# def regrid_dataset_3x3(ds, check_dims=False):
#     """
#     Input: Xarray dataset
#             - Must contain (lat, lon)

#     Output: Regridded Xarray dataset
#             to 3 deg lat lon

#     """
#     # Rename
#     if check_dims:
#         ds = data.check_dimensions(ds, ignore_dim='lon')
#     # build regridder
#     ds_out = xr.Dataset(
#         {
#             'lat': (['lat'], np.arange(-90, 93, 3)),
#             'lon': (['lon'], np.arange(0,360, 3))
#         }
#     )

#     regridder = xe.Regridder(ds, ds_out, "bilinear")
#     ds_new = regridder(ds)

#     # verify that the result is the same as regridding each variable one-by-one
#     for k in ds.data_vars:
#         print(k, ds_new[k].equals(regridder(ds[k])))

#     print('Regridding and checks complete. Dataset ready.')
#     return ds_new

#=================================================================================================

if __name__ == '__main__':

    # Set var list
    variables = ['ta', 'ua', 'va']
    # variables = ['ua', 'va']

    # Set time of day for program echo
    from datetime import datetime
    now = datetime.now().strftime("%H:%M:%S")
    print("Current Time =", now)

    for var in variables:

        # reset time for counter
        now = datetime.now().strftime("%H:%M:%S")

        print(f'[{now}]: Importing paths for MIROC6 {var}...')
        # pylint: disable=line-too-long
        files = glob.glob(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/daily/{var}/pdSST-futArcSIC/MIROC6/*')

        for count, item in enumerate(files):

            # reset current time
            now = datetime.now().strftime("%H:%M:%S")

            print(f'[{now}]: Opening ensemble member {var}: {count+1}')
            dataset = xr.open_mfdataset(item)
            dataset = dataset[[f'{var}']]

            print('Starting regridding function...')
            dataset = regrid_dataset_3x3(dataset, check_dims=True)

            print('Saving dataset...')
            dataset.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/pdSST-futArcSIC_3x3/MIROC6_3x3/{var}/{os.path.basename(item)}')

        print(f'Variable {var} finished.')

    print('PROGRAM COMPLETE (MIROC6).')
