"""
    This is a script for processing raw data (more specifically raw PAMIP data)
"""

from pathlib import Path
import xarray as xr
import logging
import pdb
import functions.data_wrangling as data

import warnings
warnings.filterwarnings("ignore", category=xr.SerializationWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def check_file_corruption(file_path):
    """Check if a NetCDF file is corrupt."""
    try:
        dataset = xr.open_dataset(str(file_path))
        dataset.close()
        return None  # File is not corrupt
    except (OSError, ValueError) as e:
        logging.error(f'Corrupt file found: {file_path} - {e}')
        return file_path  # Return the corrupt file path

def write_corrupt_files(exp, variable, bad_files, raw_dir):
    """Write the list of corrupt files to a text file in a 'corrupt_files' folder within raw_path."""
    corrupt_dir = raw_dir / 'corrupt_files'
    corrupt_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    file_name = f'{exp}_{variable}_corrupt-files.txt'
    output_path = corrupt_dir / file_name

    with open(output_path, mode='w+', encoding="utf-8") as file:
        if bad_files:
            for bad_file in bad_files:
                ensemble_member = str(bad_file)
                file.write(ensemble_member + '\n')
            logging.info(f'Corrupt files recorded in {output_path}.')
        else:
            file.write('No corrupt files found.\n')
            logging.info(f'No corrupt files found for experiment {exp} and variable {variable}.')

def process_file(file_path, variable, output_file_path):
    """Process an individual NetCDF file."""
    try:

        # Open dataset and standardize data
        dataset = xr.open_dataset(str(file_path))  # Convert Path to string

        # Handle variable-specific adjustments
        if variable == 'ua':
            if 'U' in dataset.data_vars:
                dataset = dataset.rename({'U': 'ua'})
            elif 'u' in dataset.data_vars:
                dataset = dataset.rename({'u': 'ua'})
        elif variable == 'epfy':
            if 'EPY' in dataset.data_vars:
                dataset = dataset.rename({'EPY': 'epfy'})
            if 'EPYpe' in dataset.data_vars:
                dataset = dataset.rename({'EPYpe': 'epfy'})
                
        if not variable in dataset.data_vars:
            raise ValueError(f'File {file_path} does not have {variable}, it has {dataset.data_vars}')

        # ensure vars and dims are now named correctly
        dataset = data.data_checker1000(dataset, check_vars=False)
        # subset data to remove spinup
        if 'ens_ax' in dataset.dims:
            dataset = dataset.sel(time=slice('2000-06', '2001-05'))
        else:
            logging.error(f'Dataset has ensemble members in time coordinate.')


        # now save dataset
        dataset.to_netcdf(output_file_path)
        logging.info(f'File {file_path} processed and saved successfully.')

    except Exception as e:
        logging.error(f'Error processing file {file_path}: {e}')
        return file_path  # Return the problematic file path

    return None  # No errors






def process_experiment(exp, variable, raw_path, save_dir):
    """Process all files for a given experiment and variable."""
    path = raw_path / exp / variable
    save_path = save_dir / exp / variable
    
    # obtain list of all files for specified experiment and variable
    file_list = sorted(path.rglob('*.nc'))

    if not file_list:
        logging.warning(f'No files found for {exp}/{variable}.')
        return []

    bad_files = []
    for file_path in file_list:
        model_name = file_path.parent.name
        output_save_path = save_path / model_name
        output_save_path.mkdir(exist_ok=True, parents=True)
        output_file_path = output_save_path / f'{file_path.name}'
        if output_file_path.exists():
            # logging.info(f'File already exists: {output_file_path}. Skipping processing.')
            continue

        corrupt_file = check_file_corruption(file_path)
        if corrupt_file is not None:  # Explicitly check if corrupt_file is not None
            bad_files.append(corrupt_file)
        else:
            error_file = process_file(file_path, variable, output_file_path)
            if error_file is not None:
                bad_files.append(error_file)

    write_corrupt_files(exp, variable, bad_files, raw_path)
    return bad_files
    
    

if __name__ == '__main__':
    
    # specify file paths
    raw_dir = Path('/home/links/ct715/data_storage/PAMIP/raw_data/monthly')
    save_dir = Path('/home/links/ct715/data_storage/PAMIP/monthly')
    
    # specify variables and experiments
    exp_list = ['1.1_pdSST-pdSIC', '1.6_pdSST-futArcSIC']
    var_list = ['ua', 'epfy']
    
    for exp in exp_list:
        for variable in var_list:
            logging.info(f'Processing experiment: {exp}, variable: {variable}')
            bad_files = process_experiment(exp, variable, raw_dir, save_dir)
            
            if bad_files:
                logging.error(f'Corrupt files for {exp}/{variable}: {bad_files}')