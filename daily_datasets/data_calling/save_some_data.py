import xarray as xr
import datetime as dt
import glob


variables = ['u', 'v', 't']
files = []

for var in variables:
    blah = glob.glob(f'/gws/nopw/j04/arctic_connect/xum/ECMWF/ERA5/Daily/{var}/*.nc')
    
    for element in blah:
        files.append(element)        
ds = xr.open_mfdataset(files, parallel=True, chunks={'longitude': 45})

# data manipulation
ds.attrs = {}  # for brevity, hide attributes
ds = ds.drop_vars('time_bnds')

# save ERA5 daily u,v,t dataset into ArctiConnect workspace
ds.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/era5_data/era5daily_uvt.nc')


# save DJF subset as well
ds = ds.sel(time=ds.time.dt.month.isin([12,1,2]))
ds.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/era5_data/era5daily_djf_uvt.nc')