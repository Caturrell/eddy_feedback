"""
    Isca specific functions, mostly data processing but possibily
    will include some data analysis functions if required.
    
    python /home/links/ct715/eddy_feedback/functions/isca.py
"""

import xarray as xr
import functions.data_wrangling as data
import functions.eddy_feedback as ef

import pdb
import logging
import os
import shutil
import time
from pathlib import Path

import sys
sys.path.append('/home/links/ct715/Isca/postprocessing/plevel_interpolation/scripts')
from plevel_fn import plevel_call, daily_average, join_files, two_daily_average, monthly_average

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def isca_abbreviations(exp_name):
    """
        Automatically find experiment directory name from the abbreviation
        in experiment name (exp_name).
    """
    
    # extract initial abbreviation from exp_name
    exp_type = exp_name.split('_')[0]
    exp_dict = {
        'HS': 'held-suarez',
        'PK': 'polvani-kushner'
    }
    exp_folder = exp_dict.get(exp_type)
    
    return exp_folder


def interp_isca_plevels(exp_name, base_dir=os.environ['GFDL_DATA'], avg_or_daily_list=['daily'], 
                        level_set='standard', mask_below_surface_set=' ', out_dir=None):
    """
    Processes netCDF files by interpolating variables to specified pressure levels.

    Parameters:
        exp_name (str): The experiment name.
        base_dir (Path): Base directory for input files.
        avg_or_daily_list (list): List of time averaging options (e.g., ['daily', 'monthly']).
        level_set (str): Level configuration ('standard' or other custom settings).
        mask_below_surface_set (str): Option for masking below surface values.
            - '-x ' for simulating pressure where land would be (CLARIFY)
        out_dir (Path): Output directory. Defaults to `base_dir` if not provided.

    Returns:
        None
    """
    
    # Set start time
    start_time = time.time()
    
    # Get the path for the experiment directory and calculate the number of files
    exp_dir = Path(base_dir) / exp_name
    start_file = 1
    # Count number of files in the directory
    nfiles = len(list(exp_dir.glob('run*')))  # Assumes file pattern "run*"
    
    # find array of pfull
    plev_array = _automate_pfull_values(exp_name, base_dir)
    
    logging.info(f"Starting the file processing for {exp_name}...")

    if out_dir is None:
        out_dir = Path(base_dir)
        logging.info(f"No output directory specified, using base directory: {out_dir}")

    logging.info(f"Total files to process: {nfiles}")

    plevs = {}
    var_names = {}

    if level_set == 'standard':
        logging.info("Using standard level set.")
        plevs = {
            # 'monthly': ' -p "1000 10000 25000 50000 85000 92500"',
            # '6hourly': ' -p "1000 10000 25000 50000 85000 92500"',
            'daily': f' -p "{plev_array}"'
        }
        
        var_names = {
            # 'monthly': '-a slp height',
            # '6hourly': '-a slp height',
            'daily': '-a slp height' # -a: all vars + SLP + geopot height
        }
        file_suffix = '_interp'
    else:
        raise ValueError("Unsupported level_set: only 'standard' is implemented.")

    # Processing the files
    for n in range(nfiles):
        for avg_or_daily in avg_or_daily_list:
            nc_file_in = exp_dir / f"run{n + start_file:04d}" / f"atmos_{avg_or_daily}.nc"
            nc_file_out = out_dir / exp_name / f"run{n + start_file:04d}" / f"atmos_{avg_or_daily}{file_suffix}.nc"

            if not nc_file_out.is_file():
                logging.info(f"Interpolating file: {nc_file_in.parent.name}/{nc_file_in.name}")
                print()
                try:
                    plevel_call(
                        str(nc_file_in), # needs to be str not PoxisPath
                        str(nc_file_out),
                        var_names=var_names[avg_or_daily],
                        p_levels=plevs[avg_or_daily],
                        mask_below_surface_option=mask_below_surface_set
                    )
                    logging.info(f"Successfully interpolated and saved to: {nc_file_out}")
                except Exception as e:
                    logging.error(f"Error during interpolation of {nc_file_in}: {e}")
            else:
                logging.info(f"File already exists, skipping: {nc_file_out}")

    logging.info('File processing completed.')
    logging.info(f'Execution time: {time.time() - start_time} seconds')
    
    

# HELPER FUNCTIONS
#-----------------
    
def _automate_pfull_values(EXPERIMENT, path_to_dir, pfull_var='pfull'):
    """
    Processes the pressure levels from an Xarray dataset.

    Parameters:
        EXPERIMENT (str): The experiment name.
        path_to_dir (str): The path to the directory containing the experiment data.
        pfull_var (str): The name of the pressure level variable in the dataset (default: 'pfull').

    Returns:
        str: A space-separated string of the processed pressure levels.
    """
    # Construct the file path
    file_path = Path(path_to_dir) / EXPERIMENT / 'run0001' / 'atmos_daily.nc'

    # Open the dataset
    dataset = xr.open_dataset(file_path)

    if pfull_var not in dataset:
        raise ValueError(f"Variable '{pfull_var}' not found in the dataset.")

    # Extract the pfull variable
    pfull = dataset[pfull_var]

    # Round to nearest whole number and multiply by 100
    processed_levels = (pfull.round().astype(int) * 100).values

    # Convert to space-separated string
    levels_string = ' '.join(map(str, processed_levels))

    return levels_string
    
#==================================================================================================

#-----------------------------
# SINGULAR DATASET MANAGEMENT
#-----------------------------
    

def standardise_isca_data(exp_name, path_to_dir, file_name='atmos_daily.nc'):
    
    """
        Take raw isca data and convert it to a single netCDF file
        in my required format.
    """
    
    # automatically set desired folder destination based on experiment acronym
    exp_folder = isca_abbreviations(exp_name=exp_name)
    
    # import Path and direct to required folder
    path = Path(path_to_dir)
    exp_path = path / exp_name
    atmos_daily_path = exp_path.glob(f'run*/{file_name}')
    dest_folder = path / exp_folder / 'processed_simulations'
    dest_folder.mkdir(parents=True, exist_ok=True)
    save_netCDF = path / exp_folder / f'{exp_name}_original.nc'
    
    if save_netCDF.exists():
        logging.info(f'Raw Isca data already processed: {save_netCDF}')
    else:
        # open all data into one large dataset
        logging.info('Open all datatsets into one Xarray dataset...')
        dataset = xr.open_mfdataset(
            atmos_daily_path,
            parallel=True,
            chunks={'time': 30}
        )
        
        # run data through my data checker (to ensure dims are named consistently)
        dataset = data.data_checker1000(dataset)
        
        # save dataset
        logging.info('Datasets open and data processed. Saving as one netCDF file...')
        dataset.to_netcdf(save_netCDF)
        
        # move raw data to processed folder
        logging.info('Dataset saved. Moving raw data to processed_simulations folder.')
        shutil.move(exp_path, dest_folder)
    
    logging.info('standardise_isca_data() complete.')
    
    
#------------------------------------------------
    
def create_EFP_components_dataset(exp_name, path_to_dir):
    
    """
        Creates a netCDF file that contains u, v, t, ubar;
        epfy, epfz, divFy, divFz.
    """
    
    # automatically set desired folder destination based on experiment acronym
    exp_folder = isca_abbreviations(exp_name=exp_name)
    
    PATH = Path(path_to_dir)
    dataset_name = f'{exp_name}_original.nc'
    path_to_dataset = PATH / exp_folder 
    
    # open dataset and choose desired variables
    ds = xr.open_mfdataset(
        path_to_dataset / dataset_name,
        parallel=True,
        chunks={'time': 30}
    )
    
    # choose desired variables
    var_list = ['ucomp', 'vcomp', 'temp']
    ds_new = ds[var_list]
    ds_new = data.data_checker1000(ds_new, check_vars=True)
    
    # Calculate EP fluxes
    logging.info('Dataset open and initial checks complete. Calculating EP fluxes...')
    ds_new = ef.calculate_epfluxes_ubar(ds_new)
    
    # save dataset
    logging.info('EP fluxes calculated. Now saving dataset...')
    ds_new.to_netcdf(path_to_dataset / f'{exp_name}_epf.nc')
    
    
#==================================================================================================

#--------------------------
# LARGE DATASET MANAGEMENT
#--------------------------

def calculate_divFy_then_make_dataset(exp_name, path_to_dir, use_interp=True):
    
    # automatically set desired folder destination based on experiment acronym
    exp_folder = isca_abbreviations(exp_name=exp_name)
    
    if use_interp:
        atmos_daily_file = 'atmos_daily_interp.nc'
    else:
        atmos_daily_file = 'atmos_daily.nc'
    
    # import Path and direct to required folder
    path = Path(path_to_dir)
    exp_path = path / exp_name
    atmos_daily_path = list(exp_path.iterdir())
    atmos_daily_path.sort()
    atmos_daily_path = atmos_daily_path[1:]
    
    # Destination folders
    dest_folder = path / exp_folder / 'processed_simulations'
    dest_folder.mkdir(parents=True, exist_ok=True)
    save_netCDF_path = path / exp_folder / f'{exp_name}-nc'
    save_netCDF_path.mkdir(parents=True, exist_ok=True)
    
    # open each dataset, calculate EP fluxes and save in new file
    for item in list(atmos_daily_path):
        
        logging.info(f'Processing: {item.name}')
        ds = xr.open_mfdataset(
            item / atmos_daily_file,
            parallel=True,
            chunks={'time': 30}
        )
        
        # choose desired variables
        var_list = ['ucomp', 'vcomp', 'temp']
        ds_new = ds[var_list]
        ds_new = data.data_checker1000(ds_new)
        
        # calculate EP fluxes and save dataset
        ds_new = ef.calculate_epfluxes_ubar(ds_new)
        ds_new.to_netcdf(save_netCDF_path / f'{exp_name}_uvt_epf_{item.name}.nc')
        
    logging.info(f'All datasets processed. Now moving raw data to: {dest_folder}')
    shutil.move(exp_path, dest_folder)
    
    logging.info(f'calculate_divFy_then_make_dataset() complete.')
    
        

def save_multiple_epf_into_one_dataset(exp_name, path_to_dir):
    
    # automatically set desired folder destination based on experiment acronym
    exp_folder = isca_abbreviations(exp_name=exp_name)
    
    # import Path and direct to required folder
    path = Path(path_to_dir)
    exp_path = path / exp_name
    atmos_daily_path = exp_path.glob(f'run*/{exp_name}_uvt_epf_run*.nc')
    # post-processing paths
    dest_folder = path / exp_folder / 'processed_simulations'
    dest_folder.mkdir(parents=True, exist_ok=True)
    save_netCDF = path / exp_folder / f'{exp_name}_epf.nc'

    # open into one dataset and save it
    logging.info('Opening into one large dataset and saving...')
    ds = xr.open_mfdataset(
        atmos_daily_path,
        parallel=True,
        chunks={'time': 30}
    )
    ds.to_netcdf(save_netCDF)
    





#==================================================================================================
    
# if name==main for a singular experiment
# if __name__ == '__main__':
#     # set variables
#     logging.info('Starting program.')
    
#     # set path and experiment name
#     path_to_dir=Path('/home/links/ct715/data_storage/isca')
#     EXPERIMENT_NAME = input('Insert experiment name to be processed:')
    
#     # INTERPOLATE: plevels
#     logging.info(f'Initialising interp_isca_plevels():')
#     interp_isca_plevels(EXPERIMENT_NAME)
    
    
#     # INDIVIDUAL: calculate EP fluxes first, then save dataset
#     logging.info(f'Initialising calculate_divFy_then_make_dataset():')
#     calculate_divFy_then_make_dataset(EXPERIMENT_NAME, path_to_dir)
#     # logging.info(f'Function complete. Initialising save_multiple_epf_into_one_dataset():')
#     # save_multiple_epf_into_one_dataset(EXPERIMENT_NAME, path_to_dir)
    
    
#     # # ONE LARGE DATASET: standardise raw data into one large dataset
#     # logging.info('Processing raw data. Initiating standardise_isca_data()...')
#     # standardise_isca_data(exp_name=EXPERIMENT_NAME, path_to_dir=path_to_dir)
#     # # subset dataset and calculate EP fluxes
#     # logging.info('Calculating EP fluxes and creating new datasets.')
#     # create_EFP_components_dataset(exp_name=EXPERIMENT_NAME, path_to_dir=path_to_dir)
    
    
#     logging.info('Program complete.')
    
    
# if name == main for multiple datasets
if __name__ == '__main__':
    
    
    # set Isca data path and automatically find experiments
    logging.info('Starting program.')
    path_to_dir=Path('/home/links/ct715/data_storage/isca')
    disca_list = list(path_to_dir.iterdir())
    
    exp_list = []
    for item in disca_list:
        exp_name = item.name
        name_split = exp_name.split('_')
        
        if name_split[0] == 'HS':
            exp_list.append(exp_name)
        else:
            pass
    
    # set path and experiment name
    for item in exp_list:
        
        logging.info(f'Processing experiment: {item}')
        EXPERIMENT_NAME = item
        
        # INTERPOLATE: plevels
        logging.info(f'Initialising interp_isca_plevels():')
        interp_isca_plevels(EXPERIMENT_NAME)
        
        
        # INDIVIDUAL: calculate EP fluxes first, then save dataset
        logging.info(f'Initialising calculate_divFy_then_make_dataset():')
        calculate_divFy_then_make_dataset(EXPERIMENT_NAME, path_to_dir)
        logging.info(f'Completed experiment: {item}')
        
    logging.info('Program complete.')