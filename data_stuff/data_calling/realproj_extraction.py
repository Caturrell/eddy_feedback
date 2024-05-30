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
    extract_path = f'/gws/nopw/j04/realproj/users/alwalsh/PAMIP/HadGEM3_N96/pdSST-pdSIC/r{i+1:03}i1p1f1/bj524a.pm*.nc'
    PATH = '/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/ua/HadGEM3-GC31-LL/temp_files/.'

    # extract data from Mass
    print('Extracting files...')
    os.system('cp '+extract_path+' '+PATH)

    # open data using Xarray and manipulate data before saving as one dataset
    ds = xr.open_mfdataset('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/ua/HadGEM3-GC31-LL/temp_files/*.nc',
                            combine='nested')
    ua = ds.u.rename({'t': 'time', 'p_1': 'level', 'latitude_1': 'lat', 'longitude_1': 'lon'})

    print('Data manipulation complete. Saving dataset...')
    ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/ua/HadGEM3-GC31-LL/ua_Amon_HadGEM3-GC31-MM_pdSST-pdSIC_r{i+1}.nc')

    print(f'Ensemble member {i+1} complete.')
    # delete temporary files
    os.system('rm -rf /gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/ua/HadGEM3-GC31-LL/temp_files/*')

os.system('rmdir /gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/ua/HadGEM3-GC31-LL/temp_files')

print('Program complete.')
