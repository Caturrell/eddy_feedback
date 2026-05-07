import xarray as xr
import os

from functions.data_wrangling import data_checker1000

src_file = '/disca/share/sit204/jra_55/1958_2016/height_daily/atmos_daily_together.nc'
out_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/hgt_daily'

os.makedirs(out_dir, exist_ok=True)

ds = xr.open_dataset(src_file, chunks={'time': 365})
ds = ds.rename({'var7': 'hgt'})

years = range(1958, 2017)

for year in years:
    print(f'Processing {year}...')
    ds_year = ds.sel(time=str(year))
    ds_year = data_checker1000(ds_year)
    out_path = os.path.join(out_dir, f'{year}_hgt_JRA55.nc')
    ds_year.to_netcdf(out_path)
    print(f'  Saved: {out_path}')

print('Done.')
