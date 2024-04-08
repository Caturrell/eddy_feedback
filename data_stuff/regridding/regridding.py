import numpy as np
import xarray as xr
import xesmf as xe
import glob 
import os 

import sys 
sys.path.append('/home/users/cturrell/documents/eddy_feedback')
import functions.data_wrangling as data

#======================================================================================================================================

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

    print('Checks complete. Dataset ready.')
    
    return ds_new 

    
    
    
    
#======================================================================================================================================


## JASMIN SERVERS
    
if __name__ == '__main__':

    print('Importing paths for AWI-CM-1-1-MR va...')
    files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/daily/ta/pdSST-pdSIC/AWI-CM-1-1-MR/*')
    
    
    for count, item in enumerate(files):
        
        print(f'Opening ensemble member: {count}')
        ds = xr.open_mfdataset(item)
        ds = ds[['ta']]
    
        print('Starting regridding function...')
        ds = regrid_dataset_3x3(ds, check_dims=True)
        
        print('Regrid complete. Saving dataset...')
        ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/AWI-CM-1-1-MR_3x3/ta/{os.path.basename(files[count])}')
        
    print('Program complete.')
        
    
    
    
    