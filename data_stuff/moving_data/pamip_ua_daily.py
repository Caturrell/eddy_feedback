import shutil
import xarray as xr
from pathlib import Path

source_path = Path('/gws/nopw/j04/realproj/users/alwalsh/PAMIP/HadGEM3_N96')
source_exp_list = ['pdSST-futArcSIC', 'pdSST-futArcSIC-QBOE-nc', 'pdSST-pdSIC', 'pdSST-pdSIC-QBOE-nc']

dest_path = Path('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/day/raw_hadll_ua_day')

for exp in source_exp_list:
    for i in range(100):
        # Set files to be moved
        ensemble_member = f'r{i+1:03}i1p1f1'
        path = source_path / exp / ensemble_member
        files = list(path.glob('*.pe*.nc'))
        
        # Set destination folder
        end_path = dest_path / exp / ensemble_member
        end_path.mkdir(parents=True, exist_ok=True)  # Ensure destination folder exists
        
        # Copy files to destination folder
        for file in files:
            shutil.copy2(file, end_path)
            print(f'Copied {file} to {end_path}')