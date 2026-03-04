import logging
import numpy as np
import os
from tqdm import tqdm
import xarray as xar
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to Python path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import SIT_eddy_feedback_functions as eff

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()  # Remove all existing handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class FileValidationCache:
    """Cache for tracking validated NetCDF files to avoid re-checking them."""
    
    def __init__(self, cache_file=None):
        """
        Initialize the validation cache.
        
        Parameters
        ----------
        cache_file : str or Path, optional
            Path to the cache JSON file. If None, uses default location.
        """
        if cache_file is None:
            # Save cache in the same directory as this script
            script_dir = Path(__file__).parent
            cache_file = script_dir / 'ipsl_validation_cache.json'
        self.cache_file = Path(cache_file)
        self.validated_files = self._load_cache()
        
    def _load_cache(self):
        """Load the cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                logging.info(f'Loaded validation cache with {len(cache_data)} entries from {self.cache_file}')
                return cache_data
            except Exception as e:
                logging.warning(f'Failed to load cache file: {e}. Starting with empty cache.')
                return {}
        return {}
    
    def _save_cache(self):
        """Save the cache to disk."""
        try:
            # Create parent directory if it doesn't exist
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.validated_files, f, indent=2)
            logging.debug(f'Saved validation cache with {len(self.validated_files)} entries')
        except Exception as e:
            logging.warning(f'Failed to save cache file: {e}')
    
    def is_validated(self, filepath):
        """
        Check if a file has been previously validated.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the file to check.
            
        Returns
        -------
        bool
            True if file is in cache and still exists with same modification time.
        """
        filepath = str(Path(filepath).resolve())
        
        if filepath not in self.validated_files:
            return False
        
        # Check if file still exists
        if not os.path.isfile(filepath):
            # File was deleted, remove from cache
            del self.validated_files[filepath]
            self._save_cache()
            return False
        
        # Check if modification time matches (file hasn't been modified since validation)
        cached_mtime = self.validated_files[filepath].get('mtime')
        current_mtime = os.path.getmtime(filepath)
        
        if cached_mtime is None or abs(current_mtime - cached_mtime) > 1.0:
            # File was modified, needs re-validation
            return False
        
        return True
    
    def mark_validated(self, filepath):
        """
        Mark a file as validated in the cache.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the validated file.
        """
        filepath = str(Path(filepath).resolve())
        
        if os.path.isfile(filepath):
            self.validated_files[filepath] = {
                'mtime': os.path.getmtime(filepath),
                'validated_at': datetime.now().isoformat()
            }
            self._save_cache()
    
    def clear_entry(self, filepath):
        """
        Remove a file from the validation cache.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the file to remove from cache.
        """
        filepath = str(Path(filepath).resolve())
        if filepath in self.validated_files:
            del self.validated_files[filepath]
            self._save_cache()


# Initialize global validation cache
validation_cache = FileValidationCache()


def is_valid_netcdf(filepath, use_cache=True):
    """
    Return True if file exists and can be opened and fully read without error.
    A full load is performed to catch corruption in data blocks, not just headers.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF file to validate.
    use_cache : bool, optional
        If True, use the validation cache to skip re-checking previously validated files.
        Default is True.
    
    Returns
    -------
    bool
        True if file is valid, False otherwise.
    """
    if not os.path.isfile(filepath):
        return False
    
    # Check cache first if enabled
    if use_cache and validation_cache.is_validated(filepath):
        logging.debug(f'File validation cached (skipping check): {filepath}')
        return True
    
    # Perform actual validation
    try:
        with xar.open_dataset(filepath) as ds:
            ds.load()
        
        # File is valid, add to cache
        if use_cache:
            validation_cache.mark_validated(filepath)
            logging.debug(f'File validated and cached: {filepath}')
        
        return True
    except Exception as e:
        logging.warning(f'File appears corrupt and will be regenerated: {filepath} ({e})')
        
        # Remove from cache if it was there
        if use_cache:
            validation_cache.clear_entry(filepath)
        
        return False


#==============================================================================================================
# IPSL-CM6A-LR SPECIFIC SETUP - Using pre-extracted yearly chunks
#==============================================================================================================

logging.info("="*70)
logging.info("IPSL-CM6A-LR Processing - Using Pre-extracted Yearly Chunks")
logging.info("="*70)

exp_type = 'cmip6'
force_recalculate = False
subtract_annual_cycle = True
level_type = '6hrPlevPt'

# IPSL-specific paths
base_dir_output = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
model_name = 'IPSL-CM6A-LR'
experiment = '1850_2015'

# Location of pre-extracted yearly chunks
yearly_chunk_dir = f'{base_dir_output}/{model_name}/{experiment}/{level_type}/yearly_data'

logging.info(f"Model: {model_name}")
logging.info(f"Experiment: {experiment}")
logging.info(f"Yearly chunks directory: {yearly_chunk_dir}")

# Find all yearly chunk files
if not os.path.isdir(yearly_chunk_dir):
    logging.error(f"Yearly chunk directory not found: {yearly_chunk_dir}")
    raise FileNotFoundError(f"Directory not found: {yearly_chunk_dir}")

# Find all yearly chunk files (ua and va)
ua_chunk_files = sorted([f for f in os.listdir(yearly_chunk_dir) 
                         if f.endswith('_IPSL-CM6A-LR_historical_r1i1p1f1_uv.nc')])

if not ua_chunk_files:
    logging.error(f"No yearly chunk files found in {yearly_chunk_dir}")
    raise FileNotFoundError("No yearly chunk files found")

logging.info(f"Found {len(ua_chunk_files)} yearly chunk files")

# Extract start and end dates from filenames
start_month_list_by_files = []
end_month_list_by_files = []

for chunk_file in ua_chunk_files:
    # Filename format: YYYYMMDD_YYYYMMDD_IPSL-CM6A-LR_historical_r1i1p1f1_uv.nc
    parts = chunk_file.split('_')
    start_date = parts[0]
    end_date = parts[1]
    start_month_list_by_files.append(start_date)
    end_month_list_by_files.append(end_date)

logging.info(f"Processing {len(start_month_list_by_files)} yearly chunks")
logging.info(f"First chunk: {start_month_list_by_files[0]} to {end_month_list_by_files[0]}")
logging.info(f"Last chunk: {start_month_list_by_files[-1]} to {end_month_list_by_files[-1]}")

# Create files dictionary to match expected format
files = {
    'ua': [os.path.join(yearly_chunk_dir, f) for f in ua_chunk_files],
    'va': [os.path.join(yearly_chunk_dir, f) for f in ua_chunk_files]  # Same files contain both ua and va
}

# Set up time range
start_month = start_month_list_by_files[0][:4]  # First year
end_month = end_month_list_by_files[-1][:4]     # Last year

# Variable configuration
var_name_dict = {
    'ua': {'data_type': '6hrPlevPt'},
    'va': {'data_type': '6hrPlevPt'},
}

logging.info(f"Time range: {start_month} to {end_month}")
logging.info("="*70)

#==============================================================================================================
# Main processing loop
#==============================================================================================================

force_ep_flux_recalculate = False

omega = 2.*np.pi/86400.
a0 = 6371000.
do_individual_plots = True
do_individual_corr_plots = False
do_big_TEM_plot = False
do_heatmap_correlations_plot = False
do_eof_plots = False

monthly_too = True

logging.info(f'Now processing {model_name}')

try:
    # Set up output directories
    plot_dir = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    yearly_data_dir = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/yearly_data'
    if not os.path.isdir(yearly_data_dir):
        os.makedirs(yearly_data_dir)

    logging.info(f'Output directory: {yearly_data_dir}')
    
    #-------------------------------------------------
    ## MAIN LOOP FOR PROCESSING EACH YEARLY FILE ##
    #-------------------------------------------------
    
    for year_idx, start_date_val in tqdm(enumerate(start_month_list_by_files)):    

        ## SET UP OUTPUT FILE NAMES ##
        end_date_val = end_month_list_by_files[year_idx]

        output_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_epflux.nc'
        output_day_av_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_daily_averages.nc'
        output_month_av_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_monthly_averages.nc'

        # Get files for this year (both ua and va are in the same file)
        files_for_year = [files['ua'][year_idx]]
        
        logging.info(f'Processing files for {model_name}; year starting {start_date_val} to {end_date_val}')

        # Do the calculation if output files are missing, corrupt, or recalculation is forced.
        # is_valid_netcdf() opens and fully loads each file to detect both header and data-block corruption.
        # The validation cache speeds up this check by remembering previously validated files.
        if force_ep_flux_recalculate or not is_valid_netcdf(output_file) or not is_valid_netcdf(output_day_av_file):
            logging.info(f'Opening model data files: {files_for_year}')
            time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
            dataset = xar.open_mfdataset(files_for_year, decode_times=time_coder,
                                    parallel=False, join='inner')

            # Check and set up plev properly
            if np.all(dataset.plev.diff('plev')>0.):
                pfull_slice = slice(100., 1000.)
            else:
                pfull_slice = slice(1000., 100.)

            # Check and convert plev to hPa if needed
            if 'units' in dataset['plev'].attrs.keys():
                if dataset['plev'].attrs['units']=='Pa':
                    dataset['plev'] = dataset['plev']/100.
            elif dataset['plev'].max().values>1000.:
                    dataset['plev'] = dataset['plev']/100.                    

            do_pfull_slice = True

            # Check there are enough plev levels after slicing 
            # and reopen without plev slicing if not
            if dataset.plev.sel(plev=pfull_slice).shape[0]<3:
                dataset.close()
                logging.info('Re-opening model data files due to too few pfull levels')
                dataset = xar.open_mfdataset(files_for_year, decode_times=time_coder,
                                        parallel=False)
                do_pfull_slice = False
                if dataset.plev.shape[0]<3:
                    orig_plev = dataset.dims['plev'].values
                    new_plev  = np.zeros(3)
                    new_plev[:orig_plev.shape[0]] = orig_plev
                    new_plev[orig_plev.shape[0]:] = np.zeros(3-orig_plev.shape[0]) + np.nan

            # Rename CMIP6 variables to expected names
            dataset = dataset.rename({
                'ua': 'ucomp',
                'va': 'vcomp',
                'plev': 'pfull',
            })
            
            if 'ta' in dataset.keys():
                dataset = dataset.rename({
                    'ta': 'temp',
                })                                                

            if 'units' in dataset['pfull'].attrs.keys():
                if dataset['pfull'].attrs['units']=='Pa':
                    dataset['pfull'] = dataset['pfull']/100.
            elif dataset['pfull'].max().values>1000.:
                    dataset['pfull'] = dataset['pfull']/100.
            
            # Check for inconsistent chunk sizes
            chunk_size_var_dict = {}
            ds_coords = [val for val in dataset.coords.keys()]
            ds_vars = [val for val in dataset.variables.keys() if val not in ds_coords and 'bnds' not in val and 'bounds' not in val]
            for var_name in ds_vars:
                chunk_size_var_dict[var_name] = dataset[var_name].chunksizes
            all_chunks_equal = [chunk_size_var_dict[var_name] == chunk_size_var_dict[ds_vars[0]] for var_name in ds_vars]
            if not np.all(all_chunks_equal):
                dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0], 'lat':dataset.lat.shape[0], 'lon':dataset.lon.shape[0]})

            # Check for udt_rdamp
            if 'udt_rdamp' in dataset.data_vars.keys():
                include_udt_rdamp = True
            else:
                include_udt_rdamp = False    

            if np.all(dataset.pfull.diff('pfull')>0.):
                pfull_slice = slice(100., 1000.)
            else:
                pfull_slice = slice(1000., 100.)

            if do_pfull_slice:
                dataset = dataset.sel(pfull=pfull_slice)

            # Rechunk if needed
            chunk_size_var_dict = {}
            ds_coords = [val for val in dataset.coords.keys()]
            ds_vars = [val for val in dataset.variables.keys() if val not in ds_coords and 'bnds' not in val and 'bounds' not in val]
            for var_name in ds_vars:
                chunk_size_var_dict[var_name] = dataset[var_name].chunksizes
            all_chunks_equal = [chunk_size_var_dict[var_name] == chunk_size_var_dict[ds_vars[0]] for var_name in ds_vars]
            if not np.all(all_chunks_equal):
                dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0], 'lat':dataset.lat.shape[0], 'lon':dataset.lon.shape[0]})

            if min(chunk_size_var_dict[ds_vars[0]]['pfull'])<3:
                logging.warning('Rechunking pfull to allow edge-order 2 vert deriv')
                dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0]})

            logging.info(f'\nThis dataset: {model_name} has {dataset.pfull.shape[0]} plevels, {dataset.lat.shape[0]} lat points, {dataset.time.shape[0]} time points and {dataset.lon.shape[0]} longitude points\n')
            
            # Add temp if missing
            if 'temp' not in dataset.keys():
                dataset['temp'] = xar.zeros_like(dataset['ucomp']) + np.nan

            #-------------------------------------------------
            ## CALCULATE EP FLUXES ##
            #-------------------------------------------------       

            epflux_ds = eff.ep_flux_calc(dataset, output_file, force_ep_flux_recalculate, include_udt_rdamp, omega, a0)
            
            dataset = xar.merge([dataset, epflux_ds])
            logging.info(f'Finished calculated EP fluxes for {model_name} and {output_file} and merged dataset.')

            dataset, duplicates_found = eff.check_for_duplicate_times(dataset)

            if duplicates_found:
                dataset, duplicates_found_2 = eff.check_for_duplicate_times(dataset)
                assert(not duplicates_found_2)
        else:
            dataset = None
            epflux_ds = None

        logging.info('Calculating daily averages...')
        dataset_daily = eff.daily_average(dataset, output_day_av_file, force_ep_flux_recalculate, monthly_too=monthly_too, monthly_output_file=output_month_av_file)

        dataset_daily.close()

        if epflux_ds is not None:
            epflux_ds.close()

        if dataset is not None:
            dataset.close()

    logging.info("="*70)
    logging.info(f"✅ Successfully processed all yearly chunks for {model_name}")
    logging.info("="*70)

except Exception as e:
    logging.error(f'❌ Failed for {model_name} with error:')
    logging.error(e)
    import traceback
    traceback.print_exc()
    raise