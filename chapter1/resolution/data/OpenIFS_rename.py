"""
Rename and relocate OpenIFS data.
"""

import xarray as xr
from pathlib import Path
import logging

import pdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Base directory path
BASE_PATH = Path('/home/links/ct715/data_storage/data_import/OpenIFS/')

# List of folders and their file counts
data_folders = {
    'T1279_epf_exp11_ens100': 100,
    'T1279_U_exp11_ens100': 100,
    'T159_epf_exp11_ens300': 300,
    'T159_U_exp11_ens300': 300
}

# Loop over each folder and process the files
for folder, file_count in data_folders.items():
    folder_path = Path(BASE_PATH) / folder  # Construct folder path
    logging.info(f"Processing folder: {folder_path}")

    for i in range(1, file_count + 1):
        # Define the path to the current netCDF file
        nc_dir = folder_path / f"E{i:03d}" / "outdata" / "oifs" / "00001" 
        nc_path = list(nc_dir.iterdir())[0]

        if nc_dir.exists():
            logging.info(f"Found file: {nc_path}")
            
            # Load the dataset (example operation)
            try:
                ds = xr.open_dataset(nc_path)
            except Exception as e:
                logging.error(f"Error opening dataset {nc_path}: {e}")
                continue
            
            # Save the dataset to a new location (example operation)
            new_filename = f"{folder}_r{i:03d}_200006-200105.nc"
            new_path = BASE_PATH / "processed" / new_filename
            new_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                ds.to_netcdf(new_path)
                logging.info(f"Saved file to: {new_path}")
            except Exception as e:
                logging.error(f"Error saving file to {new_path}: {e}")
        else:
            logging.warning(f"File not found: {nc_path}")
