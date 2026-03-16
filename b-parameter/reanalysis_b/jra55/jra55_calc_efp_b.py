import xarray as xar
# import aostools.climate as aoscli
import numpy as np
import os
import pdb
import xcdat
import logging
import calendar


import SIT_functions.SIT_eddy_feedback_functions as eff
import SIT_functions.SIT_eddy_plotting_functions as epf


logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()  # Remove all existing handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

exp_type='jra55'
        

if exp_type=='jra55':
    base_dir = '/home/links/ct715/'
    base_dir_6hourly = '/disca/share/sit204/jra_55/1958_2016_6hourly_data_efp/full_6hourly_snapshots'
    exp_name = 'jra55'

    start_month = 1979
    end_month   = 2016
    level_type = '6hourly'
    subtract_annual_cycle = True

    if level_type=='interp':
        files = [f'/home/links/ct715/data_storage/reanalysis/jra55_daily/jra55_uvtw.nc']
    else:
        files = [f'{base_dir_6hourly}/{var_name}/atmos_6hourly_together.nc' for var_name in ['omega', 'ucomp', 'temp', 'vcomp']]

        # file_prefix_dict = {'temp':'anl_p25.011_tmp',
        #                     'omega':'anl_p25.039_vvel',
        #                     'ucomp':'anl_p25.033_ugrd',
        #                     'vcomp':'anl_p25.034_vgrd'}
        # files=[]
        # for var_name in file_prefix_dict.keys():
        #     file_prefix=file_prefix_dict[var_name]
        #     for year_val in range(start_month, end_month+1):    
        #         for month_val in range(1,13):   
        #             end_month_day = calendar.monthrange(year_val, month_val)[1]
                    
        #             file_in = f'{base_dir_6hourly}/{var_name}/{file_prefix}_{year_val}{month_val:02d}0100_{year_val}{month_val:02d}{end_month_day}18.nc'     

        #             files = files + [file_in]
else:
    raise NotImplementedError(f'no valid exp type configuresd for {exp_type}')

omega = 2.*np.pi/86400.
a0 = 6371000.
do_individual_plots = False
do_individual_corr_plots = False
do_big_TEM_plot = False
do_heatmap_correlations_plot = False
do_efp_annual_cycle_plot = True
do_eof_plots = True
do_power_spectrum = True

force_ep_flux_recalculate = False
force_efp_recalculate = False
force_anom_recalculate = False
force_eof_recalculate = False

plot_dir = f'./{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/'

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

yearly_data_dir = '/disca/share/sit204/jra_55/1958_2016_6hourly_data_efp'

output_file = f'./{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/epflux.nc'
output_day_av_file = f'./{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/daily_averages.nc'
output_efp_file = f'./{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/efp.nc'
output_eof_file = f'./{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/EOF.nc'
output_anom_file= f'./{exp_name}_sit_plots/{start_month}_{end_month}/{level_type}/anoms.nc'

pfull_slice = slice(100., 800.)


if level_type=='interp':
    logging.info('opening original data')
    time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
    dataset = xar.open_mfdataset(files, decode_times=time_coder,
                                parallel=True)
    logging.info('COMPLETE')

    if exp_type=='jra55':

        if level_type=='interp':
            logging.info('opening OLD JRA-55 dataset to grab time bounds')
            old_dataset = xar.open_mfdataset(['/disca/share/sit204/jra_55/1958_2016/atmos_daily_uvtw.nc'], decode_times=time_coder,
                                    parallel=True, chunks={'time': 50})    
            
            dataset['time_bnds'] = old_dataset['time_bnds']    
            logging.info('finished adding OLD JRA-55 dataset time bounds')    

            dataset = dataset.rename({
                'u':'ucomp',
                'v':'vcomp',
                't':'temp',
                'level':'pfull',
            })    

            dataset = dataset.sel(time=slice(f'{start_month}-01-01', f'{end_month}-12-31'))
        elif level_type=='6hourly':    
            dataset = dataset.rename({
                'var33':'ucomp',
                'var34':'vcomp',
                'var11':'temp',
                'var39':'omega',
                'plev':'pfull',
            })        
            if dataset['pfull'].attrs['units']=='Pa':
                dataset['pfull'] = dataset['pfull']/100.        


        dataset = dataset.sel(lat=slice(87.5,-87.5))

    if 'udt_rdamp' in dataset.data_vars.keys():
        include_udt_rdamp=True
    else:
        include_udt_rdamp=False    

    epflux_ds = eff.ep_flux_calc(dataset, output_file, force_ep_flux_recalculate, include_udt_rdamp, omega, a0)

    logging.info('merging dataset')
    dataset = xar.merge([dataset, epflux_ds])
    logging.info('FINISHED merging dataset')


    # dataset = eff.daily_average(dataset, output_day_av_file, force_ep_flux_recalculate)

elif level_type=='6hourly':
    dataset = eff.read_daily_averages(yearly_data_dir, start_month, end_month)
else:
    raise NotImplementedError('Not sure hot to handle this')

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

efp_output_ds = eff.efp_calc(output_efp_file, force_efp_recalculate, dataset, vars_to_correlate, exp_type, season_month_dict)

individual_plot_list = ['ep1', 'ep2', 'div1', 'div2', 'fvbarstar', 'vbarstar', 'omegabarstar', 'dudphi', '1oacosphi_dudphi', 'dudp', 'total_tend', 'delta_ubar_dt', 'dthdp_bar', 'inv_dthdp_bar']


if do_individual_plots:
    epf.individual_plots(dataset, individual_plot_list, plot_dir)

if do_individual_corr_plots:
    epf.individual_corr_plots(efp_output_ds, dataset, vars_to_correlate, plot_dir, season_month_dict)

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


#Plan for next analysis is to recreate figure 4 from Lorenz and Hartmann, i.e. power spectra and autocorrelation of the zonal wind metric and then the eddy forcing metrics, and then figure 5 (cross correlation). Which leads which? And then maybe do this for more terms in the TEM equation. Do any of the eddy terms lead eachother etc? 


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