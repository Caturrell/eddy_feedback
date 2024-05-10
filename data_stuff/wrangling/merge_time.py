""" 
    Script used to merge TIME dimension on netCDF files.
"""
# pylint: disable=line-too-long

import xarray as xr

variables = ['ua', 'epfy']

for item in variables:

    print('Program starting...')

    if item == 'ua':
        ID = 'Amon'
    else:
        ID = 'EmonZ'
    
    print(f'Table ID for {item}: {ID}')

    for i in range(300):

        print(f'Starting new merge ({item}).')
        # paths for each ensemble
        path1 = f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/{item}/HadGEM3-GC31-MM_2files/{item}_{ID}_HadGEM3-GC31-MM_pdSST-pdSIC_r{i+1}i1p1f2_gn_200004-200012.nc'
        path2 = f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/{item}/HadGEM3-GC31-MM_2files/{item}_{ID}_HadGEM3-GC31-MM_pdSST-pdSIC_r{i+1}i1p1f2_gn_200101-200105.nc'

        # make new dataset and save it
        dataset = xr.open_mfdataset([path1, path2], combine='nested', concat_dim='time')
        dataset.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/{item}/HadGEM3-GC31-MM/{item}_Amon_HadGEM3-GC31-MM_pdSST-pdSIC_r{i+1}i1p1f2_gn_200004-200105.nc')
        print(f'Ensemble member {i+1} completed ({item}_{ID}).')

    print(f'Variable {item}_{ID} completed.')

print('Script complete.')
