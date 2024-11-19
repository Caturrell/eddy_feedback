import xarray as xr
from pathlib import Path

# set paths
PATH = Path('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino_pdSST-pdSIC/ua/HadGEM3-GC31-MM/u-ca809-nc')

# loop through each ensemble member
for i in range(150):
    
    # set path to files
    loop_path = PATH / f'r{i+1:03}i1p1f1' 
    member_file = loop_path.glob('ca809*.nc')
    
    # open dataset
    ds_member = xr.open_mfdataset(
        member_file,
        parallel=True,
        combine='nested',
        concat_dim='time'
    )
    
    # rename x_wind (ua)
    ds_member.rename(
        {'x_wind': 'ua'}
    )
    ds_member = ds_member.sortby('time')
    
    # save dataset
    save_name = f'ua_Amon_HadGEM3-GC31-MM_pdSST-pdSIC_elnino_r{i+1:03}i1p1f1_gn_200006-200105.nc'
    save_path = PATH / save_name
    ds_member.to_netcdf(save_path)