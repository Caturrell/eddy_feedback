""" 
    Script used to pull in data from Mass, combine in time, then iterate for each ensemble.
"""
# pylint: disable=line-too-long

# import subprocess, sys
import os
import xarray as xr


### Do some magic with bash here and call in data

for i in range(100):

    # set file path for each iteration
    extract_path = f'/gws/nopw/j04/realproj/users/alwalsh/PAMIP/HadGEM3_N96/pdSST-pdSIC_elnino_QBOE-nc/r{i+1:03}i1p1f1/*.pm*.nc'
    PATH = '/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino-QBOE_pdSST-pdSIC/ua/temp_files/.'

    # extract data from Mass
    print('Extracting files...')
    os.system('cp '+extract_path+' '+PATH)

    # open data using Xarray and manipulate data before saving as one dataset
    ds = xr.open_mfdataset('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino-QBOE_pdSST-pdSIC/ua/*.nc',
                            combine='nested')
    
    # rename variables to my usual naming convention and discard of spin up
    ua = ds.u.rename({'t': 'time', 'p_1': 'level', 'latitude_1': 'lat', 'longitude_1': 'lon'})
    ua = ua.sel(time=slice('2000-06', '2001-05'))

    # save manipulated dataset
    print('Data manipulation complete. Saving dataset...')
    ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino-QBOE_pdSST-pdSIC/ua/ua_Amon_HadGEM3-GC31-LL_pdSST-pdSIC_elnino_QBOE-nc_r{i+1:03}_gn_200006-200105.nc')

    print(f'Ensemble member {i+1} complete.')
    # delete temporary files
    os.system('rm -rf /gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino-QBOE_pdSST-pdSIC/ua/temp_files/*')

os.system('rmdir /gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino-QBOE_pdSST-pdSIC/ua/temp_files/.')

print('Program complete.')
