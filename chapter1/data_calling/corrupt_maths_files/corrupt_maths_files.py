import os
from pathlib import Path
import xarray as xr
import logging

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
raw_path = Path('/home/links/ct715/data_storage/PAMIP/monthly')
exp_names = ['1.1_pdSST-pdSIC', '1.6_pdSST-futArcSIC']

for exp in exp_names:
    path = raw_path / exp / 'ua'
    files_rglob = path.rglob('*.nc')
    file_list = list(files_rglob)

    # Empty list for corrupt file paths
    bad_files = []

    for item in file_list:
        logging.info(f'Opening file: {item}')
        # Attempt to open individual file
        try:
            dataset = xr.open_dataset(str(item))  # Ensure Path object is converted to string
            dataset.close()
        except (OSError, ValueError) as e:
            bad_files.append(item)
            logging.error(f'Corrupt file found: {item} - {e}')
        else:
            logging.info(f'File {item} opened successfully.')

    # Write corrupt files to a text file
    file_name = f'{exp}_ua_corrupt-files.txt'
    output_path = Path('/home/links/ct715/eddy_feedback/chapter1/data_calling/corrupt_maths_files') / file_name

    with open(output_path, mode='w+', encoding="utf-8") as file:
        for bad_file in bad_files:
            ensemble_member = os.path.basename(bad_file)
            file.write(ensemble_member + '\n')

    # Check if the output file is empty
    if not bad_files:
        with open(output_path, mode='w+', encoding="utf-8") as file:
            file.write('No corrupt files found.')
        logging.info(f'No corrupt files found for experiment {exp}.')
    else:
        logging.info(f'Corrupt files recorded in {output_path}.')

logging.info('Program completed.')
