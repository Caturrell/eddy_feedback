""" 
    A script for finding and recording corrupt netCDF files.
"""
# pylint: disable=line-too-long

import os
import glob
import xarray as xr

# Load in file paths
files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/1.6_pdSST-futArcSIC/ua/HadGEM3-GC31-MM_2files/*.nc')

# Empty list for corrupt file paths
bad_files = []

for item in files:
    print(f'Opening file: {item}')
    # attempt to open individual file
    try:
        dataset = xr.open_dataset(item)
        dataset.close()
    # if fails, append to list and print Exception
    except OSError as e1:
        bad_files.append(item)
        print(f'Corrupt file found: {e1}')
    except ValueError as e2:
        bad_files.append(item)
        print(f'Corrupt file found: {e2}')
    else:
        print('No error found in this file.')

# PRINT BAD_FILES OUTPUT TO TXT FILE
# file = open('HadGEM3-MM_corrupt-files.txt', mode='w', encoding="utf-8")
# for item in bad_files:
#     file.write(item+'\n')
# file.close()

FILE_NAME = 'HadGEM3-MM_futArc_ua_corrupt-files.txt'
with open(FILE_NAME, mode='w+', encoding="utf-8") as file:
    for item in bad_files:
        ensemble_member = os.path.basename(item)
        file.write(ensemble_member+'\n')

# check if file is empty, then leave message if so
path_size = os.path.getsize(f'/home/users/cturrell/documents/eddy_feedback/data_stuff/wrangling/{FILE_NAME}')
if path_size == 0:
    with open(FILE_NAME, mode='w+', encoding="utf-8") as file:
        file.write('No corrupt files found.')

print('Program completed.')
