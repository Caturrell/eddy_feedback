""" 
    A script for finding and recording corrupt netCDF files.
"""
# pylint: disable=line-too-long

import os
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
    # if fails, append to list and print Exception
    except OSError:
        bad_files.append(item)
        print('Corrupt file found.')
    else:
        print('No error found in this file.')

# PRINT BAD_FILES OUTPUT TO TXT FILE
# file = open('CanESM5_corrupt-files.txt', mode='w', encoding="utf-8")
# for item in bad_files:
#     file.write(item+'\n')
# file.close()

with open('CanESM5_corrupt-files.txt', mode='w+', encoding="utf-8") as file:
    for item in bad_files:
        file_name = os.path.basename(item)
        file.write(file_name+'\n')

print('Program completed.')
