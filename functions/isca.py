"""
    Isca specific functions, mostly data processing but possibily
    will include some data analysis functions if required.
    
    python /home/links/ct715/eddy_feedback/functions/isca.py
"""

import xarray as xr
import functions.data_wrangling as data
import functions.eddy_feedback as ef

import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def standardise_isca_data(exp_name=None, path_to_dir='/home/links/ct715/data_storage/isca'):
    
    """
        Take raw isca data and convert it to a single netCDF file
        in my required format.
    """
    
    # automatically set desired folder destination based on experiment acronym
    exp_type = exp_name.split('_')[0]
    exp_dict = {
        'HS': 'held-suarez',
        'PK': 'polvani-kushner'
    }
    exp_folder = exp_dict.get(exp_type)
    
    # import Path and direct to required folder
    path = Path(path_to_dir)
    exp_path = path / exp_name
    atmos_daily_path = exp_path.glob('run*/atmos_daily.nc')
    dest_folder = path / exp_folder / 'processed_simulations'
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    
    # open all data into one large dataset
    logging.info('Open all datatsets into one Xarray dataset...')
    dataset = xr.open_mfdataset(
        atmos_daily_path,
        parallel=True,
        chunks={'time': 30}
    )
    
    # run data through my data checker (to ensure dims are named consistently)
    logging.info('Datasets open. Processing data...')
    dataset = data.data_checker1000(dataset)
    
    # save dataset
    logging.info('Data processed. Saving as one netCDF file...')
    dataset.to_netcdf(path / exp_folder / f'{exp_name}_original.nc')
    
    # move raw data to processed folder
    logging.info('Dataset saved. Moving raw data to processed_simulations folder.')
    shutil.move(exp_path, dest_folder)
    
    logging.info('standardise_isca_data() complete.')
    
    
#------------------------------------------------
    
def create_EFP_components_dataset(exp_name=None, path_to_dir='/home/links/ct715/data_storage/isca'):
    
    """
        Creates a netCDF file that contains u, v, t, ubar;
        epfy, epfz, divFy, divFz.
    """
    
    # automatically set desired folder destination based on experiment acronym
    exp_type = exp_name.split('_')[0]
    exp_dict = {
        'HS': 'held-suarez',
        'PK': 'polvani-kushner'
    }
    exp_folder = exp_dict[exp_type]
    
    PATH = Path(path_to_dir)
    dataset_name = f'{exp_name}_original.nc'
    path_to_dataset = PATH / exp_folder 
    
    # open dataset and choose desired variables
    logging.info('Opening dataset and subsetting data...')
    ds = xr.open_mfdataset(
        path_to_dataset / dataset_name,
        parallel=True,
        chunks={'time': 30}
    )
    
    # choose desired variables
    var_list = ['ucomp', 'vcomp', 'temp']
    ds = ds[var_list]
    ds = data.data_checker1000(ds, check_vars=True)
    
    # Calculate EP fluxes
    logging.info('Variables chosen and renamed. Calculating EP fluxes...')
    ds = ef.calculate_epfluxes_ubar(ds)
    
    # save dataset
    logging.info('EP fluxes calculated. Now saving dataset...')
    ds.to_netcdf(path_to_dataset / f'{exp_name}_EP.nc')
    
    
#==================================================================================================
    
if __name__ == '__main__':
    
    # set variables
    logging.info('Starting program.')
    EXPERIMENT_NAME = 'HS_T42_100y_60delh'
    
    # standardise raw data into one large dataset
    logging.info('Processing raw data. Initiating standardise_isca_data()...')
    standardise_isca_data(exp_name=EXPERIMENT_NAME)
    
    # subset dataset and calculate EP fluxes
    logging.info('Calculating EP fluxes and creating new datasets.')
    create_EFP_components_dataset(exp_name=EXPERIMENT_NAME)
    
    logging.info('Program complete.')