import logging
import numpy as np
from SIT_search_for_hist import find_ensemble_list_multi_var

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()  # Remove all existing handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

exp_type='cmip6'

if exp_type=='cmip6':
    
    
    ## SEARCH VARIABLES ##
    #----------------------
    end_year_required = 2015       # fix end point for all models
    min_years_required = 15       # minimum acceptable record length
    
    
    
    force_recalculate=False
    
    subtract_annual_cycle=True
    level_type='6hrPlevPt'

    mip_id = 'CMIP'
    base_dir_badc = f'/badc/cmip6/data/CMIP6/{mip_id}/'
    base_dir_output = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'

    version_name='latest'
    experiment = 'historical'


    # Variables we want to look for
    var_name_dict = {
        'ua':{'data_type':'6hrPlevPt'},
        'va':{'data_type':'6hrPlevPt'},
        # 'zg':{'data_type':'6hrPlevPt'},
        'zg':{'data_type':'day'},
    }

    logging.info('finding all available data')
    
    models_by_var = find_ensemble_list_multi_var(base_dir_badc, var_name_dict, experiment)
    available_models_dict_to_use = models_by_var['zg']
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
            elif 'r1i1p1f3' in ens_list:
                member_choice = 'r1i1p1f3'
                logging.warning(f'-    Using f3 for {model_name}')
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

logging.info('\n--- Data availability for good models ---')
for model_name in sorted(good_model_list):
    start = start_month_dict[model_name]
    end   = end_month_dict[model_name]
    # sum years across the zg file list as a proxy for total coverage
    zg_files = one_member_files_dict[model_name]['files']['zg']
    start_years = one_member_files_dict[model_name]['start_date']['zg']
    end_years   = one_member_files_dict[model_name]['end_date']['zg']
    total_years = int(end_years[-1][:4]) - int(start_years[0][:4]) + 1
    logging.info(f'  {model_name:<30}  {start} – {end}  ({total_years} years,  {len(zg_files)} files)')