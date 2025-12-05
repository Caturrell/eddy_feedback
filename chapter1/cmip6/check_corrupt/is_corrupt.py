import os
import xarray as xr
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_corrupt(filepath):
    """Check if a NetCDF file is corrupt by trying to open it with xarray."""
    try:
        with xr.open_dataset(filepath) as ds:
            ds.load()
        return False
    except Exception as e:
        logging.debug(f"Error opening {filepath}: {e}")
        return True

def check_netcdf_files(root_dir, log_file='corrupt_files.txt'):
    corrupt_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.nc'):
                full_path = os.path.join(dirpath, file)
                logging.info(f"Checking file: {full_path}")
                if is_corrupt(full_path):
                    logging.warning(f"Corrupt file found: {full_path}")
                    corrupt_files.append(full_path)

    if corrupt_files:
        with open(log_file, 'w') as f:
            for filepath in corrupt_files:
                f.write(filepath + '\n')
        logging.info(f"Finished. {len(corrupt_files)} corrupt files logged in '{log_file}'.")
    else:
        logging.info("Finished. No corrupt files found.")

if __name__ == '__main__':
    # Set your directory path here
    root_dir = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit/30y/daily_averages'  # <-- Change this to your path
    check_netcdf_files(root_dir)
