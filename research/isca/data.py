import xarray as xr

import sys 
sys.path.append('/home/links/ct715/eddy_feedback/')
import functions.eddy_feedback as ef 

# open dataset
print('Opening dataset...')
ds = xr.open_mfdataset('/home/links/ct715/data_storage/isca/polvani-kushner/PK_T42_100y_60delh/run*/*.nc',
                       parallel=True, chunks={'time':360})

print('Dataset open. Calculating EP fluxes...')
# subset data and calculate ep fluxes
ds = ef.calculate_epfluxes_ubar(ds)
# take final 90 days of data set
ds = ds.isel(time=slice(360, None))
# flip pressure levels
ds = ds.sel(level=slice(None,None,-1))

print('Calculations complete. Saving dataset...')
ds.to_netcdf('/home/links/ct715/data_storage/isca/polvani-kushner/PK_T42_100y_60delh_EP.nc')