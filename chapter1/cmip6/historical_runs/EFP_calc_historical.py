import xarray as xar
# import aostools.climate as aoscli
import numpy as np
import os
import pdb
import xcdat
import logging
import calendar
import SIT_eddy_feedback_functions as eff
import SIT_eddy_plotting_functions as epf
from SIT_search_for_hist import find_ensemble_list, find_ensemble_list_multi_var
from tqdm import tqdm

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()  # Remove all existing handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

#==============================================================================================================
# Same code as in search_cmip6_hist.py to find models and files (17/12/25)
#==============================================================================================================


## SEARCH VARIABLES ##
#----------------------
end_year_required = 2015       # fix end point for all models
min_years_required = 65        # minimum acceptable record length
efp_year_range = (1958, 2014)



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

logging.info('\n')
logging.info('='*70)
logging.info('='*70)
logging.info('Finished setting up model file lists')
logging.info('='*70)
logging.info('='*70)
logging.info('\n')


## 21 MODELS HAVE 150 YEARS OF DATA ##
# good_model_list = ['TaiESM1', 'AWI-ESM-1-1-LR', 'FGOALS-f3-L', 'CMCC-CM2-HR4', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-AerChem', 
#                    'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'MPI-ESM-1-2-HAM', 'IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA', 'KIOST-ESM', 'MIROC-ES2L', 
#                    'MIROC6', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'UKESM1-0-LL', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'GFDL-ESM4']


omega = 2.*np.pi/86400.
a0 = 6371000.
do_individual_plots = False
do_individual_corr_plots = False
do_big_TEM_plot = False
do_heatmap_correlations_plot = False
do_efp_annual_cycle_plot = False
do_eof_plots = False
do_power_spectrum = False

force_ep_flux_recalculate = False
force_efp_recalculate = False
use_500hPa_only = True
force_anom_recalculate = False
force_eof_recalculate = False

if exp_type=='cmip6':
    model_list = [key for key in one_member_files_dict.keys()]
else:
    model_list = [exp_type]



## COMPLETED MODELS
# model_list = ['TaiESM1', 'AWI-ESM-1-1-LR', 'FGOALS-f3-L', 'CMCC-CM2-HR4', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-AerChem', 
#                    'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'MPI-ESM-1-2-HAM', 'IPSL-CM6A-LR', 'IPSL-CM6A-LR-INCA', 'KIOST-ESM', 'MIROC-ES2L', 
#                    'MIROC6', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'UKESM1-0-LL', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'GFDL-ESM4']

for model_name in tqdm(sorted(good_model_list)):

    logging.info(f'Now looking at {model_name}')
    # if True:
    try:
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

        output_file = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/epflux.nc'
        output_day_av_file = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/daily_averages.nc'
        output_efp_file = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/efp.nc'
        output_eof_file = f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/EOF.nc'
        output_anom_file= f'{base_dir_output}/{model_name}/{start_month}_{end_month}/{level_type}/anoms.nc'


        if level_type=='6hourly' or level_type=='6hrPlevPt':
            dataset = eff.read_daily_averages(yearly_data_dir, start_month_list_by_files, end_month_list_by_files)
            dataset_monthly = eff.read_daily_averages(yearly_data_dir, start_month_list_by_files, end_month_list_by_files, daily_monthly='monthly')        
            # dataset = dataset_monthly

            if np.all(dataset.pfull.diff('pfull')>0.):
                pfull_slice = slice(100., 800.)
            else:
                pfull_slice = slice(800., 100.)        

        else:
            raise NotImplementedError('Not sure hot to handle this')

        if 'udt_rdamp' in dataset.data_vars.keys():
            include_udt_rdamp=True
        else:
            include_udt_rdamp=False    

        season_month_dict = {'DJF':[12,1,2,],
                        'JFM':[1,2,3],
                        'FMA':[2,3,4],
                        'MAM':[3,4,5],
                        'AMJ':[4,5,6],
                        'MJJ':[5,6,7],
                        'JJA':[6,7,8], 
                        'JAS':[7,8,9],
                        'ASO':[8,9,10],
                        'SON':[9,10,11],
                        'OND':[10,11,12],
                        'NDJ':[11,12,1],
    }

        # vars_to_correlate = ['fvbarstar', 'vbarstar_1oacosphi_dudphi', 'omegabarstar_dudp', 'div1', 'div2', 'total_tend', 'div1_QG', 'div2_QG', 'div1_QG_123', 'div1_QG_gt3']
        vars_to_correlate = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']

        efp_output_ds = eff.efp_calc(output_efp_file, force_efp_recalculate, dataset_monthly, vars_to_correlate, exp_type, season_month_dict, use_500hPa_only=use_500hPa_only, year_range=efp_year_range)

        individual_plot_list = ['div1_QG', 'div2_QG', 'fvbarstar', 'total_tend_QG']


        if do_individual_plots:
            epf.individual_plots(dataset, individual_plot_list, plot_dir)

        if do_individual_corr_plots:
            epf.individual_corr_plots(efp_output_ds, dataset, vars_to_correlate, plot_dir, season_month_dict, use_500hPa_only)

        if do_heatmap_correlations_plot:
            epf.heatmap_tem_plot(vars_to_correlate, plot_dir, efp_output_ds, season_month_dict, use_500hPa_only)

        if do_efp_annual_cycle_plot:
            epf.efp_annual_cycle(plot_dir, efp_output_ds, season_month_dict, use_500hPa_only)

        plot_title_dict = {
                'fvbarstar':r"$f \bar{v}^*$",
                'vbarstar_1oacosphi_dudphi':r"$- \bar{v}^* \frac{1}{a \cos \phi} \frac{\partial (\bar{u} \cos \phi)}{\partial \phi}$",
                'omegabarstar_dudp': r"$ - \bar{\omega}^* \frac{\partial \bar{u}}{\partial p}$",
                'div1':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$", 
                'div2':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p}$", 
                'div1_QG':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$ QG", 
                'div2_QG':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p}$ QG",    
                'div1_123':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$ 123", 
                'div2_123':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p}$ 123", 
                'div1_QG_123':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$ QG 123", 
                'div2_QG_123':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p} 123$ QG",            
                'div1_gt3':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$ gt3", 
                'div2_gt3':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p}$ gt3", 
                'div1_QG_gt3':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$ QG gt3", 
                'div2_QG_gt3':r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p} gt3$ QG",                             
                'udt_rdamp':r"$\bar{\varepsilon}_u$",
                'delta_ubar_dt':r"$\frac{\partial \bar{u}}{\partial t}$",
                'total_tend':'Total RHS',
                'total_tend_QG':'Total RHS QG',
                'ucomp':r"$\overline{u}$"}


        if do_big_TEM_plot:
            # epf.big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir)
            epf.big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir, use_qg=True)


        #Plan for next analysis is to recreate figure 4 from Lorenz and Hartmann, i.e. power spectra and autocorrelation of the zonal wind metric and then the eddy forcing metrics, and then figure 5 (cross correlation). Which leads which? And then maybe do this for more terms in the TEM equation. Do any of the eddy terms lead eachother etc? 

        if False:
            # eof_vars = ['ucomp', 'fvbarstar', 'vbarstar_1oacosphi_dudphi', 'omegabarstar_dudp', 'div1', 'div2', 'total_tend', 'div1_QG', 'div2_QG', 'div1_123', 'div1_gt3', 'div1_QG_123', 'div1_QG_gt3']
            eof_vars = ['ucomp', 'div1_QG', 'div1_QG_123', 'div1_QG_gt3']
            anom_ds = eff.calculate_anomalies(dataset, eof_vars, subtract_annual_cycle, output_anom_file, force_anom_recalculate)

            n_eofs = 3


            propogate_all_nans = True

            eof_ds = eff.eof_calc(exp_type, output_eof_file, force_eof_recalculate, dataset, pfull_slice, subtract_annual_cycle, eof_vars, n_eofs, season_month_dict, anom_ds, propogate_all_nans)

            lag_len = 40

            if do_eof_plots:
                epf.eof_plots(eof_vars, eof_ds, n_eofs, season_month_dict, lag_len, plot_dir, plot_title_dict, propogate_all_nans)

            #calculate power spectra

            # eff.power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj=False)
            # eff.power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj=True)

            # eff.power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj=False, use_qg=True)
            if do_power_spectrum:
                eff.power_spectrum_analysis(eof_ds, plot_dir, season_month_dict, use_div1_proj=True)

            b_dataset = eff.b_fit_simpson_2013(eof_ds, plot_dir, season_month_dict, use_div1_proj=True)
            epf.plot_b_annual_cycle(b_dataset, season_month_dict, plot_dir)
            #TASKS

            # 6. Have subsetted EFP calculation by season, but my values don't match the published values, with mine being much lower. I think this must be the same problem that Charlie had when calculating his own EP fluxes. 
    except Exception as e:
        logging.info(f'failed for {model_name} with reason:')
        logging.info(e)
        logging.info('continuing')