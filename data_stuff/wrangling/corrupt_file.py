""" 
    A script for finding and recording corrupt netCDF files.
"""
# pylint: disable=line-too-long

import glob
import xarray as xr

# Load in file paths
files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/ua/CanESM5/Amon/*.nc')

# Empty list for corrupt file paths
bad_files = []

for item in files:
    print(f'Opening file: {item}')
    # attempt to open individual file
    try:
        dataset = xr.open_dataset(item)
        dataset.close()
        print('No error.')
    # if fails, append to list and print Exception
    except OSError:
        bad_files.append(item)
        print('Corrupt file found.')

# PRINT BAD_FILES OUTPUT TO TXT FILE
