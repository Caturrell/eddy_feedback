import os
import xarray as xr
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for file paths
BASE_PATH = '/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino-QBOE_pdSST-pdSIC/ua'
TEMP_DIR = Path(BASE_PATH) / 'temp_files'
TARGET_DIR = Path(BASE_PATH)
os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure temp directory exists

def extract_and_process_data(ensemble_index: int):
    extract_path = f'/gws/nopw/j04/realproj/users/alwalsh/PAMIP/HadGEM3_N96/pdSST-pdSIC_elnino_QBOE-nc/r{ensemble_index:03}i1p1f1/*.pm*.nc'
    
    # Extract data
    logging.info(f'Extracting files for ensemble {ensemble_index}...')
    for file in Path(extract_path).parent.glob("*.pm*.nc"):
        shutil.copy(file, TEMP_DIR)
    
    # Open and manipulate the dataset
    logging.info(f'Opening and manipulating dataset for ensemble {ensemble_index}...')
    with xr.open_mfdataset(str(TEMP_DIR / '*.nc'), combine='nested') as ds:
        ua = ds.u.rename({'t': 'time', 'p_1': 'level', 'latitude_1': 'lat', 'longitude_1': 'lon'})
        ua = ua.sel(time=slice('2000-06', '2001-05'))
    
        # Save the manipulated dataset
        output_file = TARGET_DIR / f'ua_Amon_HadGEM3-GC31-LL_pdSST-pdSIC_elnino_QBOE-nc_r{ensemble_index:03}_gn_200006-200105.nc'
        ua.to_netcdf(output_file)
    
    logging.info(f'Ensemble member {ensemble_index} complete.')
    
    # Clean up temporary files
    for temp_file in TEMP_DIR.glob('*'):
        temp_file.unlink()
    logging.info(f'Temporary files for ensemble {ensemble_index} removed.')

def main():
    # Iterate over ensemble members
    for i in range(100):
        extract_and_process_data(i + 1)

    # Remove the temp directory after processing all files
    TEMP_DIR.rmdir()
    logging.info('Program complete.')

if __name__ == '__main__':
    main()
