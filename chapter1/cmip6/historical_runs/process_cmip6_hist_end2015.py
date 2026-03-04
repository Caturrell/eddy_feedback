import logging
import numpy as np
from SIT_search_for_hist import find_ensemble_list_multi_var
import os
from tqdm import tqdm
import xarray as xar
import xcdat
import xesmf as xe
import pdb
import json
from pathlib import Path
from datetime import datetime

import SIT_eddy_plotting_functions as epf
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
            cache_file = script_dir / 'cmip6_validation_cache.json'
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
# Same code as in search_cmip6_hist.py to find models and files (17/12/25)
#==============================================================================================================

exp_type='cmip6'

if exp_type=='cmip6':
    
    force_recalculate=False
    
    subtract_annual_cycle=True
    level_type='6hrPlevPt'

    mip_id = 'CMIP'
    base_dir_badc = f'/badc/cmip6/data/CMIP6/{mip_id}/'
    base_dir_output = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'

    version_name='latest'
    experiment = 'historical'

    end_year_required = 2015       # fix end point for all models
    min_years_required = 65       # minimum acceptable record length

    # Variables we want to look for
    var_name_dict = {
        'ua':{'data_type':'6hrPlevPt'},
        'va':{'data_type':'6hrPlevPt'},
    }

    logging.info('finding all available data')
    
    models_by_var = find_ensemble_list_multi_var(base_dir_badc, var_name_dict, experiment)
    available_models_dict_to_use = models_by_var['ua']
    logging.info('done finding all available data')


    #==============================================================================================
    # For each model, find files for one ensemble member only (e.g., r1i1p1f1)
    #==============================================================================================

    one_member_files_dict = {}
    start_month_dict = {}
    end_month_dict = {}
    dodgy_model_list = []
    weird_model_list = []

    for model_name in available_models_dict_to_use.keys():
        
        model_path = available_models_dict_to_use[model_name]['data_dir']
        ens_ids = available_models_dict_to_use[model_name]['ens_ids']
        n_files_per_ens_member = available_models_dict_to_use[model_name]['n_files_per_ens_member']
        
        logging.info(f'Processing model {model_name} at {model_path}\n')
        logging.info(f'-    Model path: {model_path}')
        logging.info(f'-    Ensemble ID: {ens_ids}')

        files_for_model = []
        n_files_required_list = []
        end_year_list = []
        n_years_available_list = []
        files_for_model_dict = {}
        start_date_for_model_dict = {}
        end_date_for_model_dict = {}

        # FIX 6: track start year per variable so we don't silently use the last loop value
        start_year_per_var = {}

        continuous_timeseries = True  # initialise before inner loop

        for var_name_to_check in var_name_dict.keys():
            ens_list = models_by_var[var_name_to_check][model_name]['ens_ids']
            grid_list = models_by_var[var_name_to_check][model_name]['grid_list']

            if len(ens_list)==1:
                member_choice = ens_list[0]
            elif 'r1i1p1f1' in ens_list:
                member_choice = 'r1i1p1f1'
            elif 'r1i1p1f2' in ens_list:
                member_choice = 'r1i1p1f2'
                logging.warning(f'-    Using f2 for {model_name}')
            else:
                raise NotImplementedError(f'Not sure which ens_id to choose for {model_name}\n\n{model_path}')

            if len(grid_list)==1:
                grid_choice = grid_list[0]
            elif 'gn' in grid_list:
                grid_choice='gn'
            elif 'gr' in grid_list:
                grid_choice='gr'
            else:
                raise NotImplementedError(f'Not sure what type of grid to choose for {model_name}')

            logging.info(f'Using {grid_choice} and ens_id={member_choice} for {model_name} and var name {var_name_to_check}')

            if member_choice in ens_list and grid_choice in grid_list:
                files_for_model_var = [file for file in models_by_var[var_name_to_check][model_name]['files'] if member_choice in file and f'/{grid_choice}/' in file]

                start_year = [file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:4] for file_name in files_for_model_var]
                end_year   = [file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:4] for file_name in files_for_model_var]

                logging.info(f'-    Years available for {var_name_to_check}: {start_year[0]} to {end_year[-1]}')

                start_date = [file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:8] for file_name in files_for_model_var]
                end_date   = [file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:8] for file_name in files_for_model_var]

                len_each_file = np.asarray(np.int32(end_year)) - np.asarray(np.int32(start_year))
                if end_date[0][4:8] == '1231':
                    len_each_file = len_each_file + 1

                # FIX 3 & 4: filter files FIRST, then recompute derived arrays from the filtered list
                filtered_indices = [i for i, ey in enumerate(end_year) if int(ey) <= end_year_required]
                files_for_model_var = [files_for_model_var[i] for i in filtered_indices]
                start_year_filtered = [start_year[i] for i in filtered_indices]
                end_year_filtered   = [end_year[i]   for i in filtered_indices]
                start_date_filtered = [start_date[i] for i in filtered_indices]
                end_date_filtered   = [end_date[i]   for i in filtered_indices]
                len_each_file       = len_each_file[filtered_indices]

                n_files_required = len(files_for_model_var)

                # FIX 4: compute any_dt on the filtered list
                any_dt = np.int32(start_year_filtered[1:]) - np.int32(end_year_filtered[0:-1])

                if np.any(any_dt > 1.):
                    logging.warning(f'There appears to be a gap in the timeseries for model {model_name}')
                    logging.warning(f'This is going to affect your calculation')
                    continuous_timeseries = False

                # FIX 2: flag dodgy models based on actual year count, not file count
                total_years = int(np.sum(len_each_file))
                if total_years < min_years_required:
                    logging.warning(f'Insufficient data for {model_name}: only {total_years} years (minimum {min_years_required})')
                    dodgy_model_list.append(model_name)

                n_files_required_list.append(n_files_required)
                # FIX 3: use filtered end_year list
                end_year_list.append(end_year_filtered[-1] if end_year_filtered else str(end_year_required))

                n_years_available_list.append(total_years)

                files_for_model_dict[var_name_to_check]      = files_for_model_var
                start_date_for_model_dict[var_name_to_check] = start_date_filtered
                end_date_for_model_dict[var_name_to_check]   = end_date_filtered

                # FIX 6: store start year per variable
                start_year_per_var[var_name_to_check] = start_year_filtered[0] if start_year_filtered else None

        if not np.all(np.asarray(end_year_list) == end_year_list[0]):
            end_year_values = [float(v) for v in end_year_list]
            end_month_dict[model_name] = str(np.int64(np.min(np.asarray(end_year_values))))
        else:
            end_month_dict[model_name] = end_year_list[-1] if end_year_list else str(end_year_required)

        # FIX 6: use the latest start year across all variables so the record is valid for all
        all_start_years = [int(v) for v in start_year_per_var.values() if v is not None]
        start_month_dict[model_name] = str(max(all_start_years)) if all_start_years else None
        if len(set(all_start_years)) > 1:
            logging.warning(f'Variables have different start years for {model_name}: {start_year_per_var}. Using latest: {max(all_start_years)}')

        n_files_per_var = [len(files_for_model_dict[v]) for v in var_name_dict.keys()]
        if not np.all(np.asarray(n_files_per_var) == n_files_per_var[0]) or not continuous_timeseries:
            weird_model_list.append(model_name)

        one_member_files_dict[model_name] = {
            'files':      files_for_model_dict,
            'start_date': start_date_for_model_dict,
            'end_date':   end_date_for_model_dict
        }

        logging.info(f'{n_years_available_list}\n')
        
else:
    raise NotImplementedError(f'no valid exp type configured for {exp_type}')


if exp_type=='cmip6':
    model_list = [key for key in one_member_files_dict.keys()]
else:
    model_list = [exp_type]

logging.info(model_list)
logging.info('dodgy models')
logging.info(np.unique(dodgy_model_list))
logging.info('weird models')
weird_model_list = [model for model in weird_model_list if model not in dodgy_model_list]
logging.info(np.unique(weird_model_list))
logging.info('good models')
good_model_list = [model for model in model_list if model not in weird_model_list and model not in dodgy_model_list]
logging.info(f'count (minimum {min_years_required} years up to {end_year_required}): {len(good_model_list)}')
logging.info(good_model_list)



#==============================================================================================================
# Now loop over good models and calculate ep fluxes
#==============================================================================================================

force_ep_flux_recalculate = False

omega = 2.*np.pi/86400.
a0 = 6371000.
do_individual_plots = True
do_individual_corr_plots = False
do_big_TEM_plot = False
do_heatmap_correlations_plot = False
do_eof_plots = False

monthly_too=True

## 21 MODELS HAVE 150 YEARS OF DATA ##
# good_model_list = ['TaiESM1', 'AWI-ESM-1-1-LR', 'FGOALS-f3-L', 'CMCC-CM2-HR4', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-AerChem', 
#                    'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'MPI-ESM-1-2-HAM', 'IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA', 'KIOST-ESM', 'MIROC-ES2L', 
#                    'MIROC6', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'UKESM1-0-LL', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'GFDL-ESM4']

# not_completed_list = ['EC-Earth3-Veg-LR', 'IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA', 'MIROC6', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 
#                               'UKESM1-0-LL', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'GFDL-ESM4']

good_model_list = ['IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA']

for model_name in sorted(good_model_list):
    

    logging.info(f'Now looking at {model_name}')
    try:
    # if True:
    
        ## SET UP CMIP6 PARAMETERS ##
        if exp_type=='cmip6':
            start_month = start_month_dict[model_name]
            end_month   = end_month_dict[model_name]
            files = one_member_files_dict[model_name]['files']
            start_month_list_by_files = one_member_files_dict[model_name]['start_date']['ua']
            end_month_list_by_files = one_member_files_dict[model_name]['end_date']['ua']   

            logging.info(f'Reading {len(files["ua"])} files for {model_name} from {start_month} to {end_month}')

            start_month_val = int(start_month)
            end_month_val = int(end_month)        

            slice_time=False
                
                
        ## SET UP OUTPUT DIRECTORIES ##

        ## base_dir_output = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/historical'
        plot_dir = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/'
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        yearly_data_dir = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/yearly_data'
        if not os.path.isdir(yearly_data_dir):
            os.makedirs(yearly_data_dir)

        logging.info(f'Output directory: {yearly_data_dir}')
        
        # #-------------------------------------------------
        # ## SPECIAL HANDLING FOR IPSL MODELS - SPLIT INTO YEARLY CHUNKS ##
        # #-------------------------------------------------
   

        # if model_name in ['IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA']:
        #     logging.info(f'IPSL model detected ({model_name}) - checking for yearly chunks')
            
        #     # Check if yearly chunks already exist
        #     yearly_chunks_exist = True
        #     yearly_chunk_files = []
            
        #     for year_idx, start_date_val in enumerate(start_month_list_by_files):
        #         end_date_val = end_month_list_by_files[year_idx]
        #         chunk_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_yearly_chunk.nc'
        #         yearly_chunk_files.append(chunk_file)
                
        #         if not os.path.isfile(chunk_file):
        #             yearly_chunks_exist = False
            
        #     # If chunks don't exist, create them
        #     if not yearly_chunks_exist:
        #         logging.info('Creating yearly chunks for IPSL model (memory-efficient approach)')
        #         time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
                
        #         # Collect ALL files across all years
        #         all_files = []
        #         for var_name_to_check in var_name_dict.keys():
        #             all_files.extend(files[var_name_to_check])
        #         all_files = list(set(all_files))  # Remove duplicates
                
        #         logging.info(f'Opening all {len(all_files)} files with dask chunking')
                
        #         try:
        #             # Open all files once with dask chunking for memory efficiency
        #             full_dataset = xar.open_mfdataset(all_files, decode_times=time_coder,
        #                                             parallel=False, join='inner',
        #                                             chunks={'time': 365})  # Chunk by ~1 year
                    
        #             # Get unique years in the dataset
        #             import numpy as np
        #             years_in_data = np.unique(full_dataset['time'].dt.year.values)
        #             logging.info(f'Years found in dataset: {years_in_data}')
                    
        #             # Process each year
        #             for year_idx, start_date_val in enumerate(start_month_list_by_files):
        #                 end_date_val = end_month_list_by_files[year_idx]
        #                 chunk_file = yearly_chunk_files[year_idx]
                        
        #                 # Skip if this chunk already exists
        #                 if os.path.isfile(chunk_file):
        #                     logging.info(f'Chunk {chunk_file} already exists, skipping')
        #                     continue
                        
        #                 # Extract year from start_date_val (format: YYYYMM)
        #                 year_start = int(start_date_val[:4])
        #                 year_end = int(end_date_val[:4])
                        
        #                 logging.info(f'Processing year {year_start} ({start_date_val} to {end_date_val})')
                        
        #                 # Select data for this year using the memory-efficient approach
        #                 if year_start == year_end:
        #                     # Single year
        #                     yearly_subset = full_dataset.sel(time=full_dataset['time'].dt.year == year_start)
        #                 else:
        #                     # Spans multiple years (rare, but handle it)
        #                     yearly_subset = full_dataset.sel(
        #                         time=(full_dataset['time'].dt.year >= year_start) & 
        #                             (full_dataset['time'].dt.year <= year_end)
        #                     )
                        
        #                 if yearly_subset.time.shape[0] > 0:
        #                     # Fix encoding for dask-backed time variable
        #                     encoding = {
        #                         'time': {
        #                             'dtype': 'float64',
        #                             'units': yearly_subset.time.encoding.get('units', 'days since 1850-01-01 00:00:00'),
        #                             'calendar': yearly_subset.time.encoding.get('calendar', 'gregorian')
        #                         }
        #                     }
                            
        #                     # Save with compute=True to trigger dask computation
        #                     logging.info(f'Saving chunk with {yearly_subset.time.shape[0]} time steps')
        #                     yearly_subset.to_netcdf(chunk_file, unlimited_dims='time', compute=True, encoding=encoding)
        #                     logging.info(f'Saved to {chunk_file}')
        #                 else:
        #                     logging.warning(f'No data found for year {year_start}')
                    
        #             full_dataset.close()
        #             logging.info('Yearly chunks created successfully')
                    
        #         except Exception as e:
        #             logging.error(f'Error creating yearly chunks: {e}')
        #             import traceback
        #             traceback.print_exc()
        #     else:
        #         logging.info('Yearly chunks already exist - skipping creation')
            
        #     # Now update the files dictionary to point to the yearly chunks
        #     logging.info('Updating file paths to use yearly chunks')
        #     for var_name in files.keys():
        #         files[var_name] = yearly_chunk_files.copy()


        #-------------------------------------------------
        ## MAIN LOOP FOR PROCESSING EACH YEARLY FILE ##
        #-------------------------------------------------
        
        for year_idx, start_date_val in tqdm(enumerate(start_month_list_by_files)):    


            ## SET UP OUTPUT FILE NAMES ##
            end_date_val = end_month_list_by_files[year_idx]

            output_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_epflux.nc'
            output_day_av_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_daily_averages.nc'
            output_month_av_file = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_monthly_averages.nc'

            files_for_year = []
            for var_name_to_check in var_name_dict.keys():
                files_for_year = files_for_year + [files[var_name_to_check][year_idx]]

            # def preprocess(ds):
            #     """Subset dataset to years 2000-2010."""
            #     return ds.sel(time=slice("1870-01-01", "1880-12-31"))
            
            logging.info(f'Processing files for {model_name}; year starting {start_date_val} to {end_date_val}')


            # Do the calculation if output files are missing, corrupt, or recalculation is forced.
            # is_valid_netcdf() opens and fully loads each file to detect both header and data-block corruption.
            # The validation cache speeds up this check by remembering previously validated files.
            if force_ep_flux_recalculate or not is_valid_netcdf(output_file) or not is_valid_netcdf(output_day_av_file):
                logging.info(f'opening model data files for {files_for_year}')
                time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
                dataset = xar.open_mfdataset(files_for_year, decode_times=time_coder,
                                        parallel=False, join='inner' )#chunks={'time': 10, 'pfull':28,})

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
                    logging.info('re-opening model data files due to too few pfull levels')
                    dataset = xar.open_mfdataset(files_for_year, decode_times=time_coder,
                                            parallel=False )#chunks={'time': 10, 'pfull':28,}) 
                    do_pfull_slice = False
                    if dataset.plev.shape[0]<3:
                        # pdb.set_trace()
                        orig_plev = dataset.dims['plev'].values
                        new_plev  = np.zeros(3)
                        new_plev[:orig_plev.shape[0]] = orig_plev
                        new_plev[orig_plev.shape[0]:] = np.zeros(3-orig_plev.shape[0]) + np.nan
                        # pdb.set_trace()

                # Check for inconsistent grids
                inconsistent_grid = False
                if dataset.lat.shape[0]==0:
                    logging.info('inconsistent lat values across files')
                    inconsistent_grid = True
                if dataset.lon.shape[0]==0:
                    logging.info('inconsistent lon values across files')
                    inconsistent_grid = True



                # If INCONSISTENT GRID, IDENTIFY ODD FILE AND REGRID TO COMMON GRID
                if inconsistent_grid:
                    logging.info('need to identify which file has an inconsistent grid')
                    ds_list = []
                    lat_list = []
                    lon_list = []
                    for file_in_year in files_for_year:
                        ds_temp = xar.open_mfdataset(file_in_year, decode_times=time_coder, parallel=False)
                        ds_list.append(ds_temp)
                        lat_list.append(ds_temp.lat.values)
                        lon_list.append(ds_temp.lon.values)

                    # does the first one match any of the others?
                    for file_idx in range(len(files_for_year)):
                        arr_lat_match_len = [lat_list_val.shape[0] == lat_list[file_idx].shape[0] for lat_list_val in lat_list]
                        arr_lon_match_len = [lon_list_val.shape[0] == lon_list[file_idx].shape[0] for lon_list_val in lon_list]        
                        if np.any(np.logical_not(arr_lat_match_len)):              
                            odd_one_out_lat_len = np.where(np.logical_not(arr_lat_match_len))[0]
                            if odd_one_out_lat_len.shape[0]==1:
                                logging.info(f'uniquely identified that it is the {odd_one_out_lat_len[0]} index file that is odd')
                                same_one_idx = np.where(arr_lat_match_len)[0][0]
                                odd_one_idx = odd_one_out_lat_len[0]

                                regridded_file_name = files_for_year[odd_one_idx].split('/')[-1]
                                final_regrid_file_name = f'{yearly_data_dir}/{regridded_file_name}'
                                if not os.path.isfile(final_regrid_file_name):
                                    logging.info('regridded file does not exist - calculating')
                                    ds_out = xar.Dataset({'lat': (['lat'], ds_list[same_one_idx].lat.values),
                                                        'lon': (['lon'], ds_list[same_one_idx].lon.values),
                                                        }
                                                    )

                                    #       ds_out.attrs = ens_mean_dataset.attrs
                                    logging.info('setting up regrid')
                                    regridder = xe.Regridder(ds_list[odd_one_idx], ds_out, 'bilinear', ignore_degenerate=True)
                                    # regridder.clean_weight_file()

                                    ds_out = regridder(ds_list[odd_one_idx])
                                    logging.info('writing regrid to file')

                                    ds_out.to_netcdf(final_regrid_file_name)
                                    ds_out.close()
                                else:
                                    logging.info('regridded file does exist - opening')

                                ok_files = [file_name for file_name in files_for_year if file_name!=files_for_year[odd_one_idx]]
                                ok_files.append(final_regrid_file_name)
                                logging.info('reopening file')
                                dataset = xar.open_mfdataset(ok_files, decode_times=time_coder,
                                            parallel=False)
                                logging.info('should now have a proper size array')
                                pass

                        else:
                            arr_lat_match = [lat_list_val == lat_list[file_idx] for lat_list_val in lat_list]
                            arr_lon_match = [lon_list_val == lon_list[file_idx] for lon_list_val in lon_list]
                            if np.where(np.asarray(arr_lat_match))[0].shape[0]>1:
                                logging.info('the first file matches at least one of the others')
                                odd_lat_idx_out = np.where(not arr_lat_match)[0]
                                odd_lon_idx_out = np.where(not arr_lon_match)[0]                            
                            raise NotImplementedError('Help')                    
                    
                logging.info('COMPLETED inconsistent grid check') 



                # set up cmip6-specific variable names and attributes
                if exp_type=='cmip6':
                    dataset = dataset.rename({
                        'ua':'ucomp',
                        'va':'vcomp',
                        'plev':'pfull',
                        })
                    
                    if 'ta' in dataset.keys():
                        dataset = dataset.rename({
                            'ta':'temp',
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


                # Not sure what udt_ramp is
                if 'udt_rdamp' in dataset.data_vars.keys():
                    include_udt_rdamp=True
                else:
                    include_udt_rdamp=False    

                if np.all(dataset.pfull.diff('pfull')>0.):
                    pfull_slice = slice(100., 1000.)
                else:
                    pfull_slice = slice(1000., 100.)

                if do_pfull_slice:
                    dataset = dataset.sel(pfull=pfull_slice)

                # Chunking stuff again? Seems a repeat of above....
                chunk_size_var_dict = {}
                ds_coords = [val for val in dataset.coords.keys()]
                ds_vars = [val for val in dataset.variables.keys() if val not in ds_coords and 'bnds' not in val and 'bounds' not in val]
                for var_name in ds_vars:
                    chunk_size_var_dict[var_name] = dataset[var_name].chunksizes
                all_chunks_equal = [chunk_size_var_dict[var_name] == chunk_size_var_dict[ds_vars[0]] for var_name in ds_vars]
                if not np.all(all_chunks_equal):
                    dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0], 'lat':dataset.lat.shape[0], 'lon':dataset.lon.shape[0]})

                if min(chunk_size_var_dict[ds_vars[0]]['pfull'])<3:
                    logging.warning('rechunking pfull to allow edge-order 2 vert deriv')
                    dataset = dataset.chunk(chunks={'pfull':dataset.pfull.shape[0]})



                logging.info(f'\nThis dataset: {model_name} has {dataset.pfull.shape[0]} plevels, {dataset.lat.shape[0]} lat points , {dataset.time.shape[0]} time points and {dataset.lon.shape[0]} longitude points\n')
                
                
                
                if 'temp' not in dataset.keys():
                    dataset['temp'] = xar.zeros_like(dataset['ucomp'])+np.nan




                #-------------------------------------------------
                ## CALCULATE EP FLUXES ##
                #-------------------------------------------------       


                # try:
                epflux_ds = eff.ep_flux_calc(dataset, output_file, force_ep_flux_recalculate, include_udt_rdamp, omega, a0)
                # except Exception as e:
                    # logging.info(f"An error occurred: {e}")
                    # pdb.set_trace()

                
                dataset = xar.merge([dataset, epflux_ds])
                logging.info(f'Finished calulated EP fluxes for {model_name} and {output_file} and merged dataset.')

                dataset, duplicates_found = eff.check_for_duplicate_times(dataset)

                if duplicates_found:
                    dataset, duplicates_found_2 = eff.check_for_duplicate_times(dataset) #run a second time to check if successful
                    assert(not duplicates_found_2) #make sure that duplicates were not found a second time, if not throw error
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


    except Exception as e:
        logging.info(f'failed for {model_name} with reason:')
        logging.info(e)
        logging.info('continuing')
