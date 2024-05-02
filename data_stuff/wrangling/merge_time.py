""" 
    Script used to merge TIME dimension on netCDF files.
"""
# pylint: disable=line-too-long

import xarray as xr

variables = ['ta', 'ua', 'va']

for item in variables:

    for i in range(100):

        print(f'Starting new merge ({item})...')
        # paths for each ensemble
        path1 = f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/daily/ua/pdSST-pdSIC/MIROC6/ua_Eday_MIROC6_pdSST-pdSIC_r{i+1}i1p1f1_gn_20000601-20001231.nc'
        path2 = f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/daily/ua/pdSST-pdSIC/MIROC6/ua_Eday_MIROC6_pdSST-pdSIC_r{i+1}i1p1f1_gn_20010101-20010531.nc'

        # make new dataset and save it
        dataset = xr.open_mfdataset([path1, path2], combine='nested', concat_dim='time')
        dataset.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/daily/ua/pdSST-pdSIC/MIROC6_merged/ua_Eday_MIROC6_pdSST-pdSIC_r{i+1}i1p1f1_gn_20000601-20010531.nc')
        print(f'Ensemble member {i+1} completed ({item}).')

    print(f'Variable {item} completed.')

print('Script complete.')
