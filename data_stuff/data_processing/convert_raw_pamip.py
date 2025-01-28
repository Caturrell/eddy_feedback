import xarray as xr
from pathlib import Path
import logging
import pdb

import functions.data_wrangling as data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("corrupt_files.log"),
        logging.StreamHandler()
    ]
)

# Load in file paths
raw_path = Path('/home/links/ct715/data_storage/PAMIP/raw_monthly')
save_path = Path('/home/links/ct715/data_storage/PAMIP/monthly')
exp_names = ['1.1_pdSST-pdSIC', '1.6_pdSST-futArcSIC']
var_names = ['ua', 'epfy']


for exp in exp_names:
    for variable in var_names:
        path = raw_path / exp / variable # probably need to specify models
        files_rglob = path.rglob('CESM2/*.nc')
        file_list = list(files_rglob)
        file_list = file_list.sort()
        
        pdb.set_trace()

        # Empty list for corrupt file paths
        bad_files = []

        for item in file_list:
            logging.info(f'Opening file: {item}')
            # Attempt to open individual file
            try:
                
                # open dataset and standardise data
                dataset = xr.open_dataset(str(item))  # Ensure Path object is converted to string
                
                # Second, extract ua (some are U)
                model = item.parent.name
                pdb.set_trace()
                
                if model == 'HadGEM3-GC31-LL':
                    ds = ds[['u']]
                    ds = ds.rename({'t': 'time', 'p_1': 'level',
                                    'latitude_1': 'lat', 'longitude_1': 'lon'})
                    ds = ds.rename({'u': 'ua'})
                elif model == 'CESM1-WACCM-SC':
                    ds = ds.rename({'U': 'ua'})
                elif 'ua' not in ds.data_vars:
                    print(f'{model} ({exp}) has: {ds.data_vars}')
                    
                dataset = data.data_checker1000(dataset, check_vars=False)
                
                
            except (OSError, ValueError) as e:
                bad_files.append(item)
                logging.error(f'Corrupt file found: {item} - {e}')
            else:
                logging.info(f'File {item} opened successfully.')