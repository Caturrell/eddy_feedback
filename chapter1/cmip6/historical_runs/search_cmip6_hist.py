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
    
    force_recalculate=False
    
    
    subtract_annual_cycle=True
    level_type='6hrPlevPt'

    mip_id = 'CMIP'
    base_dir_badc = f'/badc/cmip6/data/CMIP6/{mip_id}/'
    base_dir_output = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/historical'

    version_name='latest'
    experiment = 'historical'


    ## WE WANT ALL YEARS AVAILABLE
    total_time_span_required = 150 #specify number of years we want to analyse
    
    
    # Variables we want to look for
    var_name_dict = {
        'ua':{'data_type':'6hrPlevPt'}, #Pt here means snapshots, whereas 6hrPlev would be 6 hourly averages (see cell_methods here https://github.com/PCMDI/cmip6-cmor-tables/blob/main/Tables/CMIP6_6hrPlevPt.json vs here https://github.com/PCMDI/cmip6-cmor-tables/blob/main/Tables/CMIP6_6hrPlev.json)
        'va':{'data_type':'6hrPlevPt'},
        # 'ta':{'data_type':'6hrPlevPt'},
    }

    logging.info('finding all available data')
    
    
    ## PRINTS OUT ALL AVAILABLE MODELS WITH DATA FOR ALL VARIABLES SPECIFIED IN var_name_dict
    models_by_var = find_ensemble_list_multi_var(base_dir_badc, var_name_dict, experiment)
    available_models_dict_to_use=models_by_var['ua']
    logging.info('done finding all available data')
    #loop over each model 
    
    
    
    #==============================================================================================
    # For each model, find files for one ensemble member only (e.g., r1i1p1f1)
    # prints out grid choice and ensemble member choice for each model and variable in var_name_dict
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
            
            # Failures: 
            ## Not sure which ens_id to choose for UKESM1-0-LL

            if len(grid_list)==1:
                grid_choice = grid_list[0]
            elif 'gn' in grid_list:
                grid_choice='gn'
            elif 'gr' in grid_list:
                grid_choice='gr'
            else:
                raise NotImplementedError(f'Not sure what type of grid to choose for {model_name}')

            logging.info(f'Using {grid_choice} and ens_id={member_choice} for {model_name} and var name {var_name_to_check}')



            ## IF ENS_ID AND GRID CHOICE ARE VALID, GET FILES
            if member_choice in ens_list and grid_choice in grid_list:
                files_for_model_var = [file for file in models_by_var[var_name_to_check][model_name]['files'] if member_choice in file and f'/{grid_choice}/' in file]

                start_year = [file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:4] for file_name in files_for_model_var]
                end_year = [file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:4] for file_name in files_for_model_var]   
                
                logging.info(f'-    Years available for {var_name_to_check}: {start_year[0]} to {end_year[-1]}')             

                start_date = [file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:8] for file_name in files_for_model_var]
                end_date = [file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:8] for file_name in files_for_model_var]                

                len_each_file = np.asarray(np.int32(end_year)) - np.asarray(np.int32(start_year))
                
                any_dt = np.int32(start_year[1:])- np.int32(end_year[0:-1])

                if end_date[0][4:8]=='1231':
                    len_each_file = len_each_file + 1

                n_files_required = np.int32(np.ceil(total_time_span_required/len_each_file[0]))
                n_files_required_float = total_time_span_required/len_each_file[0]

                continuous_timeseries = True
                if np.any(any_dt>1.):
                    logging.warning(f'there appears to be a gap in the timeseries for model {model_name}')
                    if np.any(any_dt[0:n_files_required]>1):
                        logging.warning(f'This is going to affect your calculation')
                        continuous_timeseries=False

                if n_files_required>len(files_for_model_var):
                    logging.info(f'Insufficient files for {model_name}')
                    dodgy_model_list.append(model_name)
                    n_files_required = len(files_for_model_var)

                n_files_required_list.append(n_files_required)
                end_year_list.append(end_year[n_files_required-1])

                n_years_available_list.append(np.sum(len_each_file))

                files_for_model_dict[var_name_to_check] = files_for_model_var[0:n_files_required]
                start_date_for_model_dict[var_name_to_check] = start_date[0:n_files_required]
                end_date_for_model_dict[var_name_to_check] = end_date[0:n_files_required]

        # if not np.all(np.asarray(n_files_required_list) == n_files_required_list[0]):
            # pdb.set_trace()
        if not np.all(np.asarray(end_year_list) == end_year_list[0]):
            end_year_values = [float(end_year_val) for end_year_val in end_year_list]
            end_month_dict[model_name]   = str(np.int64(np.min(np.asarray(end_year_values))))
        else:
            end_month_dict[model_name]   = end_year[n_files_required-1]

        start_month_dict[model_name] = start_year[0]

        n_files_per_var = []
        for var_name_to_check in var_name_dict.keys():
            n_files_per_var.append(len(files_for_model_dict[var_name_to_check]))
        if not np.all(np.asarray(n_files_per_var)==n_files_per_var[0]) or not continuous_timeseries:
            weird_model_list.append(model_name)

        one_member_files_dict[model_name] = {'files':files_for_model_dict, 'start_date':start_date_for_model_dict, 'end_date':end_date_for_model_dict}

        logging.info(n_years_available_list)
        
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
logging.info(f'count (minimum {total_time_span_required}): {len(good_model_list)}')
logging.info(good_model_list)