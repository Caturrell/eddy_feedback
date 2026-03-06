"""
    python /home/links/ct715/isca-efp/cmip6_tem_budget.py
"""
# This script runs the complete analysis pipeline for the isca-efp project.
# It sets up the environment, configures logging, selects model data based on experiment type,
# reads the necessary files, computes EP fluxes, eddy-feedback parameters, anomalies, EOFs,
# and then generates a series of diagnostic plots including power spectra, autocorrelations,
# and spatial patterns of the TEM (Transformed Eulerian Mean) budget terms.

import xarray as xar
import aostools.climate as aoscli  # Provides functions to compute EP fluxes and related diagnostics.
import numpy as np
import os
import pdb
import xcdat
import logging
import eddy_feedback_functions as eff  # Contains functions for computing cross-spectra, correlations, etc.
import eddy_plotting_functions as epf   # Contains plotting functions for individual plots, heatmaps, and EOF plots.
from search_for_data import find_ensemble_list, find_ensemble_list_multi_var
from tqdm import tqdm

#------------------------------------------------------------------------------
# Logging configuration
#------------------------------------------------------------------------------
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()  # Remove all existing handlers to avoid duplicate logs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Reduce verbosity for matplotlib logging.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Automatically find the user's username and configure paths accordingly.
username = os.path.expanduser("~").split("/")[-1]
if username == 'ct715':
    charlie = True
    home_path = '/home/links/ct715'
elif username == 'sit204':
    charlie = False
else:
    raise NotImplementedError(f'No configuration for {username}')

#------------------------------------------------------------------------------
# Model Choice Section
#------------------------------------------------------------------------------
# The experiment type (exp_type) determines the dataset to be used.
exp_type = 'jra55'

if exp_type == 'held_suarez':
    base_dir = '/home/links/sit204/isca_data'
    exp_name = 'held_suarez_t42_efp'
    start_month = 36
    end_month = 1236
    level_type = 'interp'
    subtract_annual_cycle = False
    if level_type == 'sigma':
        file_name = 'atmos_daily.nc'
    else:
        file_name = 'atmos_daily_interp.nc'
    files = [f'{base_dir}/{exp_name}/run{run_idx:04d}/{file_name}' for run_idx in range(start_month, end_month + 1)]
        
elif exp_type == 'isca':
    base_dir = '/home/links/sit204/isca_data'
    exp_name = 'soc_ga3_do_simple_false_cmip_o3_bucket_qflux_co2_400_mid_alb_gfort_mm_mp'
    start_month = 264
    end_month = 376    
    level_type = 'interp'
    subtract_annual_cycle = True
    if level_type == 'sigma':
        file_name = 'atmos_daily.nc'
    else:
        file_name = 'atmos_daily_interp_efp2.nc'
    files = [f'{base_dir}/{exp_name}/run{run_idx:04d}/{file_name}' for run_idx in range(start_month, end_month + 1)]

elif exp_type == 'jra55':
    if charlie:
        base_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily'
    else:
        base_dir = '/home/links/sit204/'
    exp_name = 'jra55'
    start_month = 1958
    end_month = 2016   
    level_type = 'interp'
    subtract_annual_cycle = True
    files = [f'{base_dir}/jra55_uvtw.nc']

elif exp_type == 'cmip6':
    # For CMIP6, the script uses a different file selection mechanism.
    subtract_annual_cycle = True
    level_type = 'interp'
    mip_id = 'CMIP'
    base_dir_badc = f'/badc/cmip6/data/CMIP6/{mip_id}/'
    version_name = 'latest'
    experiment = 'piControl'
    total_time_span_required = 10.0  # Number of years to analyze
    force_recalculate = True
    var_name_dict = {
        'ua': {'data_type': 'day'},
        'va': {'data_type': 'day'},
        'ta': {'data_type': 'day'},
    }
    print('finding all available data')
    models_by_var = find_ensemble_list_multi_var(base_dir_badc, var_name_dict, experiment)
    available_models_dict_to_use = models_by_var['ua']
    print('done finding all available data')
    # Loop over each model to choose one ensemble member and grid.
    one_member_files_dict = {}    
    start_month_dict = {}
    end_month_dict = {}
    for model_name in available_models_dict_to_use.keys():
        files_for_model = []
        for var_name_to_check in var_name_dict.keys():
            ens_list = models_by_var[var_name_to_check][model_name]['ens_ids']
            grid_list = models_by_var[var_name_to_check][model_name]['grid_list']
            ens_list = models_by_var[var_name_to_check][model_name]['ens_ids']
            if len(ens_list) == 1:
                member_choice = ens_list[0]
            elif 'r1i1p1f1' in ens_list:
                member_choice = 'r1i1p1f1'
            else:
                raise NotImplementedError(f'Not sure which ens_id to choose for {model_name}')
            if len(grid_list) == 1:
                grid_choice = grid_list[0]
            elif 'gn' in grid_list:
                grid_choice = 'gn'
            elif 'gr' in grid_list:
                grid_choice = 'gr'                
            else:
                raise NotImplementedError(f'Not sure what type of grid to choose for {model_name}')
            logging.info(f'Using {grid_choice} and ens_id={member_choice} for {model_name}')
            if member_choice in ens_list and grid_choice in grid_list:
                files_for_model_var = [file for file in models_by_var[var_name_to_check][model_name]['files'] if member_choice in file and f'/{grid_choice}/' in file]
                start_year = [file_name.split('_')[-1].split('.nc')[0].split('-')[0][0:4] for file_name in files_for_model_var]
                end_year = [file_name.split('_')[-1].split('.nc')[0].split('-')[1][0:4] for file_name in files_for_model_var]                
                len_each_file = np.asarray(np.int32(end_year)) - np.asarray(np.int32(start_year)) + 1
                n_files_required = np.int32(np.ceil(total_time_span_required / len_each_file[0]))
                files_for_model = files_for_model + files_for_model_var[0:n_files_required]
        start_month_dict[model_name] = start_year[0]
        end_month_dict[model_name] = end_year[n_files_required - 1]
        one_member_files_dict[model_name] = files_for_model
else:
    raise NotImplementedError(f'no valid exp type configured for {exp_type}')

#------------------------------------------------------------------------------
# Set Parameters and Run Script
#------------------------------------------------------------------------------
# Set various physical and plotting parameters.
omega = 2. * np.pi / 86400.  # Earth's angular frequency (rad/s)
a0 = 6371000.                # Earth's radius in meters
do_individual_plots = True
do_individual_corr_plots = True
do_big_TEM_plot = True
do_heatmap_correlations_plot = True
do_eof_plots = True
force_ep_flux_recalculate = False
force_efp_recalculate = False
force_anom_recalculate = False
force_eof_recalculate = False

# Determine model list based on experiment type.
if exp_type == 'cmip6':
    model_list = [key for key in one_member_files_dict.keys()]
else:
    model_list = [exp_type]

#------------------------------------------------------------------------------
# Loop Over Models and Process Data
#------------------------------------------------------------------------------
for model_name in tqdm(model_list):
    logging.info(f'Now looking at {model_name}')
    if exp_type == 'cmip6':
        start_month = start_month_dict[model_name]
        end_month = end_month_dict[model_name]
        files = one_member_files_dict[model_name]
        logging.info(f'Reading {len(files)} files for {model_name} from {start_month} to {end_month}')
    save_plot_path = f'{home_path}/eddy_feedback/chapter2/isca-efp_figures'
    plot_dir = f'{save_plot_path}/{model_name}/{start_month}_{end_month}/{level_type}/'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    # Define file paths for outputs.
    output_file = f'{save_plot_path}/{model_name}/{start_month}_{end_month}/{level_type}/epflux.nc'
    output_efp_file = f'{save_plot_path}/{model_name}/{start_month}_{end_month}/{level_type}/efp.nc'
    output_eof_file = f'{save_plot_path}/{model_name}/{start_month}_{end_month}/{level_type}/EOF.nc'
    output_anom_file = f'{save_plot_path}/{model_name}/{start_month}_{end_month}/{level_type}/anoms.nc'

    logging.info('opening model data files')
    # Use CFDatetimeCoder for proper time decoding.
    time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
    dataset = xar.open_mfdataset(files, decode_times=time_coder,
                                 parallel=True, chunks={'time': 50})
    logging.info('COMPLETE')
    
    #------------------------------------------------------------------------------
    # Special Handling for JRA55 and CMIP6 Datasets
    #------------------------------------------------------------------------------
    if exp_type == 'jra55':
        logging.info('opening OLD JRA-55 dataset to grab time bounds')
        old_dataset = xar.open_mfdataset(['/disca/share/sit204/jra_55/1958_2016/atmos_daily_uvtw.nc'], decode_times=time_coder,
                                         parallel=True, chunks={'time': 50})    
        dataset['time_bnds'] = old_dataset['time_bnds']    
        logging.info('finished adding OLD JRA-55 dataset time bounds')    
        # Rename variables to standard names.
        dataset = dataset.rename({
            'u': 'ucomp',
            'v': 'vcomp',
            't': 'temp',
            'level': 'pfull',
        })
    elif exp_type == 'cmip6':
        dataset = dataset.rename({
            'ua': 'ucomp',
            'va': 'vcomp',
            'ta': 'temp',
            'plev': 'pfull',
        })
        if dataset['pfull'].attrs['units'] == 'Pa':
            dataset['pfull'] = dataset['pfull'] / 100.
            
    #------------------------------------------------------------------------------
    # Determine Inclusion of Additional Variables and Pressure Slice
    #------------------------------------------------------------------------------
    if 'udt_rdamp' in dataset.data_vars.keys():
        include_udt_rdamp = True
    else:
        include_udt_rdamp = False    

    if np.all(dataset.pfull.diff('pfull') > 0.):
        pfull_slice = slice(100., 800.)
    else:
        pfull_slice = slice(800., 100.)

    logging.info(f'This dataset has {dataset.pfull.shape[0]} plevels, {dataset.lat.shape[0]} lat points, {dataset.time.shape[0]} time points and {dataset.lon.shape[0]} longitude points')

    #------------------------------------------------------------------------------
    # Compute EP Fluxes and Merge with Dataset
    #------------------------------------------------------------------------------
    epflux_ds = eff.ep_flux_calc(dataset, output_file, force_ep_flux_recalculate, include_udt_rdamp, omega, a0)
    logging.info('merging dataset')
    dataset = xar.merge([dataset, epflux_ds])
    logging.info('FINISHED merging dataset')

    if exp_type == 'jra55':
        dataset = dataset.sel(lat=slice(87.5, -87.5))

    vars_to_correlate = ['fvbarstar', 'div1_QG', 'div2_QG', 'total_tend_QG']
    efp_output_ds = eff.efp_calc(output_efp_file, force_efp_recalculate, dataset, vars_to_correlate, exp_type)
    individual_plot_list = ['div1_QG', 'div2_QG', 'fvbarstar', 'total_tend_QG']

    #------------------------------------------------------------------------------
    # Plotting Section
    #------------------------------------------------------------------------------
    if do_individual_plots:
        epf.individual_plots(dataset, individual_plot_list, plot_dir)
    if do_individual_corr_plots:
        epf.individual_corr_plots(efp_output_ds, dataset, vars_to_correlate, plot_dir)
    if do_heatmap_correlations_plot:
        epf.heatmap_tem_plot(vars_to_correlate, plot_dir, efp_output_ds)

    # Define a dictionary for plot titles using LaTeX formatting.
    plot_title_dict = {
        'fvbarstar': r"$f \bar{v}^*$",
        'vbarstar_1oacosphi_dudphi': r"$- \bar{v}^* \frac{1}{a \cos \phi} \frac{\partial (\bar{u} \cos \phi)}{\partial \phi}$",
        'omegabarstar_dudp': r"$ - \bar{\omega}^* \frac{\partial \bar{u}}{\partial p}$",
        'div1': r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$", 
        'div2': r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p}$", 
        'div1_QG': r"$\frac{1}{a \cos \phi} \nabla \cdot F_{\phi}$ QG", 
        'div2_QG': r"$\frac{1}{a \cos \phi} \nabla \cdot F_{p}$ QG",             
        'udt_rdamp': r"$\bar{\varepsilon}_u$",
        'delta_ubar_dt': r"$\frac{\partial \bar{u}}{\partial t}$",
        'total_tend': 'Total RHS',
        'total_tend_QG': 'Total RHS QG',
        'ucomp': r"$\overline{u}$"
    }

    if do_big_TEM_plot:
        if 'omega' in [key for key in dataset.variables.keys()]:
            epf.big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir)
        epf.big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir, use_qg=True)

    #------------------------------------------------------------------------------
    # Anomaly Calculation and EOF Analysis
    #------------------------------------------------------------------------------
    eof_vars = ['ucomp', 'fvbarstar', 'div1_QG', 'div2_QG', 'total_tend_QG']
    anom_ds = eff.calculate_anomalies(dataset, eof_vars, subtract_annual_cycle, output_anom_file, force_anom_recalculate)
    n_eofs = 3
    hemisphere_month_dict = {'n': [1, 2, 12],
                             's': [7, 8, 9]}
    propogate_all_nans = True
    eof_ds = eff.eof_calc(exp_type, output_eof_file, force_eof_recalculate, dataset, pfull_slice,
                          subtract_annual_cycle, eof_vars, n_eofs, hemisphere_month_dict, anom_ds, propogate_all_nans)
    lag_len = 40

    if do_eof_plots:
        epf.eof_plots(eof_vars, eof_ds, n_eofs, hemisphere_month_dict, lag_len, plot_dir, plot_title_dict, propogate_all_nans)

    #------------------------------------------------------------------------------
    # Power Spectrum Analysis
    #------------------------------------------------------------------------------
    # The following lines call the power spectrum analysis functions.
    # Uncomment different versions if needed.
    # eff.power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj=False)
    eff.power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj=True)
    # eff.power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj=False, use_qg=True)
    # eff.power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj=True, use_qg=True)

    #------------------------------------------------------------------------------
    # Additional Task Notes (for future work)
    #------------------------------------------------------------------------------
    # TASKS:
    # 5. Run lagged analysis on multiple different data sources (e.g., PIControl with low and high EFPs).
    # 6. Reconcile seasonal EFP calculations with published values (current values seem lower).
    # 7. Extend functionality to work when 'omega' is not present (for greater flexibility with CMIP6 data).

logging.info('Program complete.')
