import shutil
from pathlib import Path
import logging

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f'Starting program.')

# Set original data path and the new directory for data
source = Path('/gws/nopw/j04/realproj/users/alwalsh/PAMIP/MASS/ep_flux/vars_u-ca809/epfy')
destination = Path('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino_pdSST-pdSIC/epfy/HadGEM3-GC31-MM')

# Ensure the new path exists
destination.mkdir(parents=True, exist_ok=True)

for i in range(150):
    logging.info(f'Processing ensemble member {i+1:03}...')
    
    # Set old file name and the new name
    file = source / f'epfy_Amon_HadGEM3-GC31-MM_u-ca809-nc_r{i+1:03}i1p1f1_gn_200006-200105.nc'
    tmp_file = destination / f'epfy_Amon_HadGEM3-GC31-MM_u-ca809-nc_r{i+1:03}i1p1f1_gn_200006-200105.nc'
    new_name = destination / f'epfy_Amon_HadGEM3-GC31-MM_pdSST-pdSIC_elnino_r{i+1:03}i1p1f1_gn_200006-200105.nc'
    
    try:
        # Copy files over to new directory
        shutil.copy(file, destination)
        logging.info(f'File copied: {file.name} to {new_name.name}')
        
        # Rename the file
        tmp_file.rename(new_name)
        logging.info(f'File renamed: {file.name} to {new_name.name}')
    
    except Exception as e:
        logging.error(f'Error processing {file.name}: {e}')

logging.info(f'Program complete.')
