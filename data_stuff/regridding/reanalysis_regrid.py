import numpy as np
import xarray as xr
import xesmf as xe


ds = xr.open_mfdataset('/home/links/ct715/data_storage/reanalysis/jra55_daily/jra55_uvtw.nc',
                       parallel=True, chunks={'time':31})

print('Dataset opened.')

# build regridder
ds_out = xr.Dataset(
    {
        'lat': (['lat'], np.arange(-90, 93, 3)),
        'lon': (['lon'], np.arange(0,360, 3))
    }
)

regridder = xe.Regridder(ds, ds_out, "bilinear")
ds_new = regridder(ds)

print('Dataset regridded. Checking variables...')

# verify that the result is the same as regridding each variable one-by-one
for k in ds.data_vars:
    print(k, ds_new[k].equals(regridder(ds[k])))

print('Checks complete. Saving dataset...')

ds_new.to_netcdf('/home/links/ct715/data_storage/reanalysis/regridded/jra55_3x3_uvtw.nc')

print('Dataset saved. Script complete.')