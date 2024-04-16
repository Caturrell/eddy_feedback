""" 
    Test script for batch computing.
"""
import sys
import numpy as np
# import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe

# pylint: disable=wrong-import-position
sys.path.append('/home/users/cturrell/documents/eddy_feedback/')
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

#=================================================================================================

if __name__ == '__main__':

    # pylint: disable=line-too-long
    jra = xr.open_mfdataset('/gws/nopw/j04/arctic_connect/cturrell/reanalysis_data/jra55_daily/jra55_uvtw.nc')

    jra = regrid_dataset_3x3(jra)

    jra.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/reanalysis_data/jra55_daily/jra55_uvtw_3x3.nc')
