import iris
import mo_pack
from pathlib import Path
import logging
import os

# Enable the future mode for saving NetCDF files with split attributes
iris.FUTURE.save_split_attrs = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# import path and use pathlib.Path()
PATH = Path('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/elnino_pdSST-pdSIC/ua/HadGEM3-GC31-MM')
pp_files = PATH / 'u-ca809'
nc_files = PATH / 'u-ca809-nc'

logging.info('Starting conversion from .pp to .nc files')
# convert to iris cude
for i in range(150):
    
    logging.info(f'Processing member: {i+1:03}')
    
    # set pp and nc path and make directories if they don't exist
    pp_path = PATH / 'u-ca809' / f'r{i+1:03}i1p1f1'
    nc_path = PATH / 'u-ca809-nc' / f'r{i+1:03}i1p1f1'
    nc_path.mkdir(parents=True, exist_ok=True)
    
    pp_files = list(pp_path.iterdir())
    
    for item in pp_files:
        if item.suffix == '.pp':
            
            # print(item.stem)
            
            # load iris cube and save to netCDF in desired location
            cube = iris.load(item)
            iris.save(cube, nc_path / f'{item.stem}.nc')
            
    logging.info(f'Member {i+1:03} converted to nc.')
    
logging.info('Program completed.')