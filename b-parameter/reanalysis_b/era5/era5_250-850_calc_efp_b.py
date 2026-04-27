import xarray as xar
import os
import pdb
import xcdat
import logging


import functions.SIT_functions.SIT_eddy_feedback_functions as eff
import functions.SIT_functions.SIT_eddy_plotting_functions as epf


logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

exp_type = 'era5'

if exp_type == 'era5':
    exp_name = exp_type

    start_month = 1979
    end_month   = 2016
    level_type  = '6hourly'
    subtract_annual_cycle = True
else:
    raise NotImplementedError(f'no valid exp type configured for {exp_type}')

do_individual_plots          = False
do_individual_corr_plots     = False
do_big_TEM_plot              = False
do_heatmap_correlations_plot = False
do_efp_annual_cycle_plot     = True
do_eof_plots                 = True
do_power_spectrum            = False

force_efp_recalculate     = False
force_anom_recalculate    = False
force_eof_recalculate     = False

save_dir = '/home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5'
plot_dir = f'{save_dir}/{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/'

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

yearly_data_dir = '/gws/ssde/j25a/arctic_connect/cturrell/reanalysis_data/era5/6h_era5/daily_averages'

output_efp_file    = f'{save_dir}/{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/efp.nc'
output_anom_file   = f'{save_dir}/{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/anoms.nc'


include_udt_rdamp = False

# ─────────────────────────────────────────────────────────────────────────────
# Read pre-computed per-year daily-average files
# ─────────────────────────────────────────────────────────────────────────────

dataset = eff.read_daily_averages(yearly_data_dir, start_month, end_month)

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
                     'NDJ':[11,12,1]
}

vars_to_correlate = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']

efp_output_ds = eff.efp_calc(output_efp_file, force_efp_recalculate, dataset,
                              vars_to_correlate, exp_type, season_month_dict)

individual_plot_list = ['ep1', 'ep2', 'div1', 'div2', 'fvbarstar', 'vbarstar',
                        'omegabarstar', 'dudphi', '1oacosphi_dudphi', 'dudp',
                        'total_tend', 'delta_ubar_dt', 'dthdp_bar', 'inv_dthdp_bar']

if do_individual_plots:
    epf.individual_plots(dataset, individual_plot_list, plot_dir)

if do_individual_corr_plots:
    epf.individual_corr_plots(efp_output_ds, dataset, vars_to_correlate,
                               plot_dir, season_month_dict)

if do_heatmap_correlations_plot:
    epf.heatmap_tem_plot(vars_to_correlate, plot_dir, efp_output_ds, season_month_dict)

if do_efp_annual_cycle_plot:
    epf.efp_annual_cycle(plot_dir, efp_output_ds, season_month_dict)

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
    epf.big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir)
    epf.big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir, use_qg=True)

eof_vars = ['ucomp', 'div1_QG', 'div1_QG_123', 'div1_QG_gt3']

anom_ds = eff.calculate_anomalies(dataset, eof_vars, subtract_annual_cycle,
                                   output_anom_file, force_anom_recalculate)

n_eofs = 3
propogate_all_nans = True
lag_len = 40

# 3-level pressure-weighted average EOF calculation
level_configs = [
    (slice(100., 850.), [250., 500., 850.], True, '_250_500_850hPa'),
]

for pfull_slice_loop, level_subset_loop, pressure_weighted_loop, level_suffix in level_configs:

    output_eof_file = f'{save_dir}/{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/EOF{level_suffix}.nc'

    eof_plot_dir = f'{plot_dir}level{level_suffix}/'
    if not os.path.isdir(eof_plot_dir):
        os.makedirs(eof_plot_dir)

    eof_ds = eff.eof_calc(exp_type, output_eof_file, force_eof_recalculate, dataset,
                          pfull_slice_loop, subtract_annual_cycle, eof_vars, n_eofs,
                          season_month_dict, anom_ds, propogate_all_nans,
                          level_subset=level_subset_loop,
                          pressure_weighted=pressure_weighted_loop)

    if do_eof_plots:
        epf.eof_plots(eof_vars, eof_ds, n_eofs, season_month_dict, lag_len,
                      eof_plot_dir, plot_title_dict, propogate_all_nans)

    if do_power_spectrum:
        eff.power_spectrum_analysis(eof_ds, eof_plot_dir, season_month_dict,
                                    use_div1_proj=True)

    b_dataset = eff.b_fit_simpson_2013(eof_ds, eof_plot_dir, season_month_dict,
                                       use_div1_proj=True)
    epf.plot_b_annual_cycle(b_dataset, season_month_dict, eof_plot_dir)
