import xarray as xar
import xcdat
import functions.SIT_functions.SIT_aostools.climate as aoscli
import numpy as np
import os
import pdb
import logging
import gc
import matplotlib.pyplot as plt
import functions.SIT_functions.SIT_eddy_feedback_functions as eff
import functions.SIT_functions.SIT_eddy_plotting_functions as epf
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

subtract_annual_cycle = True
level_type = '6hrPlevPt'

start_month = 1979
end_month   = 2015                      # Until the end of 2014

omega = 2.*np.pi/86400.
a0 = 6371000.
do_individual_plots = False
do_individual_corr_plots = False
do_big_TEM_plot = False
do_heatmap_correlations_plot = False
do_efp_annual_cycle_plot = True
do_eof_plots = True
do_power_spectrum = False

force_efp_recalculate = False
use_500hPa_only = True
force_anom_recalculate = False
force_eof_recalculate = False

base_data_dir = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
model_list = sorted(os.listdir(base_data_dir))
possible_time_spans = ['1850_2015', '1850_2014', '1950_2015', '1950_2014']

logging.info(model_list)

for model_name in tqdm(model_list):

    logging.info(f'Now looking at {model_name}')
    try:
        # Find the time span directory for this model
        time_span = None
        for ts in possible_time_spans:
            if os.path.isdir(f'{base_data_dir}/{model_name}/{ts}/{level_type}/yearly_data'):
                time_span = ts
                break
        if time_span is None:
            logging.info(f'No valid time span directory found for {model_name}, skipping')
            continue

        model_dir = f'{base_data_dir}/{model_name}/{time_span}/{level_type}'
        yearly_data_dir = f'{model_dir}/yearly_data'

        sub_folder_variant = '250-500-850hPa_dm'
        plot_dir = f'./{sub_folder_variant}/{start_month}_{end_month}/{model_name}/{level_type}/'
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        os.makedirs(f'{model_dir}/{start_month}_{end_month}', exist_ok=True)
        output_efp_file  = f'{model_dir}/{start_month}_{end_month}/efp.nc'
        output_eof_file  = f'{model_dir}/{start_month}_{end_month}/EOF.nc'
        output_anom_file = f'{model_dir}/{start_month}_{end_month}/anoms.nc'

        dataset = eff.read_daily_averages(yearly_data_dir, start_month, end_month, exp_type='cmip6')
        dataset_monthly = eff.read_daily_averages(yearly_data_dir, start_month, end_month, daily_monthly='monthly', exp_type='cmip6')

        if np.all(dataset.pfull.diff('pfull')>0.):
            pfull_slice = slice(100., 850.)
        else:
            pfull_slice = slice(850., 100.)

        if 'udt_rdamp' in dataset.data_vars.keys():
            include_udt_rdamp = True
        else:
            include_udt_rdamp = False

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

        vars_to_correlate = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']

        efp_output_ds = eff.efp_calc(output_efp_file, force_efp_recalculate, dataset_monthly, vars_to_correlate, 
                                     'cmip', season_month_dict, use_500hPa_only=use_500hPa_only)

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
            epf.big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir, use_qg=True)

        eof_vars = ['ucomp', 'div1_QG', 'div1_QG_123', 'div1_QG_gt3']
        anom_ds = eff.calculate_anomalies(dataset, eof_vars, subtract_annual_cycle, output_anom_file, force_anom_recalculate)

        n_eofs = 3
        propogate_all_nans = True
        lag_len = 40

        eof_ds = eff.eof_calc('cmip', output_eof_file, force_eof_recalculate, dataset, pfull_slice,
                              subtract_annual_cycle, eof_vars, n_eofs, season_month_dict, anom_ds,
                              propogate_all_nans, level_subset=[250., 500., 850.],
                              pressure_weighted=True)

        if do_eof_plots:
            epf.eof_plots(eof_vars, eof_ds, n_eofs, season_month_dict, lag_len, plot_dir, plot_title_dict, propogate_all_nans)

        if do_power_spectrum:
            eff.power_spectrum_analysis(eof_ds, plot_dir, season_month_dict, use_div1_proj=True)

        b_dataset = eff.b_fit_simpson_2013(eof_ds, plot_dir, season_month_dict, use_div1_proj=True)
        epf.plot_b_annual_cycle(b_dataset, season_month_dict, plot_dir)

        # Close datasets and figures to free memory before next model
        for ds in [dataset, dataset_monthly, efp_output_ds, anom_ds, eof_ds, b_dataset]:
            try:
                ds.close()
            except Exception:
                pass
        plt.close('all')
        gc.collect()

    except Exception as e:
        logging.info(f'failed for {model_name} with reason:')
        logging.info(e)
        logging.info('continuing')
