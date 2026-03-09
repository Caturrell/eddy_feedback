import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np
import SIT_eddy_feedback_functions as eff
from tqdm import tqdm
import os
import matplotlib.patches as patches
import pdb

def individual_plots(dataset, individual_plot_list, plot_dir):

    plot_dir_indiv = f'{plot_dir}/individual_plots/'

    if not os.path.isdir(plot_dir_indiv):
        os.makedirs(plot_dir_indiv)

    for var_to_plot in individual_plot_list:
        
        logging.info(f'plotting {var_to_plot}')

        data_to_plot = dataset[var_to_plot]

        if 'time' in data_to_plot.dims:
            data_to_plot = data_to_plot.mean('time')

        if 'lon' in data_to_plot.dims:
            data_to_plot = data_to_plot.mean('lon')        

        plt.figure()
        plt.contourf(dataset['lat'], dataset['pfull'], data_to_plot, cmap='RdBu_r', levels=31)
        plt.colorbar()
        plt.ylim(1000., 0.)
        plt.savefig(f'{plot_dir_indiv}/{var_to_plot}.pdf')

    plt.close('all')

def individual_corr_plots(efp_output_ds, dataset, vars_to_correlate, plot_dir, season_month_dict, use_500hPa_only=False):

    plot_dir_corr = f'{plot_dir}/efp_plots/'
    if use_500hPa_only:
        plot_dir_corr = plot_dir_corr + '500hPa_only/'

    if not os.path.isdir(plot_dir_corr):
        os.makedirs(plot_dir_corr)

    season_list = [season_val for season_val in season_month_dict.keys()]

    all_time_season_list = season_list+['all_time']

    efp_box_edge_dict = {'n':[25., 75., 50.],
                         's':[-75., -25., 50.]}
    
    p_min = 200.
    p_max = 600.

    for time_frame in all_time_season_list:
        for hemisphere in ['n', 's']:

            plot_dir_corr2 = f'{plot_dir_corr}/{hemisphere}_hemisphere/{time_frame}'

            if not os.path.isdir(plot_dir_corr2):
                os.makedirs(plot_dir_corr2)          

            for var_to_plot_tick in vars_to_correlate:
                
                var_to_plot = f'{var_to_plot_tick}_ucomp_{hemisphere}_{time_frame}_corr'

                logging.info(f'plotting {var_to_plot}')

                data_to_plot = efp_output_ds[var_to_plot]

                if 'time' in data_to_plot.dims:
                    data_to_plot = data_to_plot.mean('time')

                if 'lon' in data_to_plot.dims:
                    data_to_plot = data_to_plot.mean('lon')        

                levels = np.linspace(-1., 1., num=40, endpoint=True)
                corr2_levels = np.linspace(0., 1., num=40, endpoint=True)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

                rect = patches.Rectangle(
                        (efp_box_edge_dict[hemisphere][0], p_max),                 # bottom left corner (x, y)
                        efp_box_edge_dict[hemisphere][2],               # width
                        p_min - p_max,                   # height (p goes top-down)
                        linewidth=2,
                        edgecolor='green',
                        facecolor='none'
                    )                
                rect2 = patches.Rectangle(
                        (efp_box_edge_dict[hemisphere][0], p_max),                 # bottom left corner (x, y)
                        efp_box_edge_dict[hemisphere][2],               # width
                        p_min - p_max,                   # height (p goes top-down)
                        linewidth=2,
                        edgecolor='green',
                        facecolor='none'
                    )          

                # plt.figure()
                cf1 = ax1.contourf(dataset['lat'], dataset['pfull'], data_to_plot, cmap='RdBu_r', levels=levels)
                fig.colorbar(cf1)
                # Create a rectangle


                # Add it to the Axes
                ax1.axes.add_patch(rect)


                ax1.set_ylim(1000., 0.)
                ax1.set_title(f"efp for {var_to_plot_tick} is {efp_output_ds[f'efp_{var_to_plot_tick}_ucomp_{hemisphere}_{time_frame}'].values}")

                cf2 = ax2.contourf(dataset['lat'], dataset['pfull'], data_to_plot**2., cmap='viridis', levels=corr2_levels)
                fig.colorbar(cf2)
                ax2.set_ylim(1000., 0.)
                ax2.set_title(f"efp for {var_to_plot_tick} is {efp_output_ds[f'efp_{var_to_plot_tick}_ucomp_{hemisphere}_{time_frame}'].values}")
                ax2.axes.add_patch(rect2)

                plt.savefig(f'{plot_dir_corr2}/{var_to_plot}_{hemisphere}.pdf')        

        plt.close('all')

def heatmap_tem_plot(vars_to_correlate, plot_dir, efp_output_ds, season_month_dict, use_500hPa_only=False):

    plot_dir_corr = f'{plot_dir}/efp_plots/'
    if use_500hPa_only:
        plot_dir_corr = plot_dir_corr + '500hPa_only/'

    if not os.path.isdir(plot_dir_corr):
        os.makedirs(plot_dir_corr)

    vars_for_heatmap = [var for var in vars_to_correlate if var!='udt_rdamp'] + ['ucomp']

    season_list = [season_val for season_val in season_month_dict.keys()]

    all_time_season_list = season_list+['all_time']

    for time_frame in all_time_season_list:
        for hemisphere in ['n','s']:

            plot_dir_corr2 = f'{plot_dir_corr}/{hemisphere}_hemisphere/{time_frame}'

            if not os.path.isdir(plot_dir_corr2):
                os.makedirs(plot_dir_corr2)    

            plt.figure(figsize=(16,16))
            n_corr_vars = len(vars_for_heatmap)

            corr_value_arr = np.zeros((n_corr_vars, n_corr_vars))

            for idx_2, var2_to_correlate in enumerate(vars_for_heatmap):
                for idx_1, var_to_correlate in enumerate(vars_for_heatmap):

                    corr_var_name = f'efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}_{time_frame}'
                    corr_value_arr[idx_2, idx_1] = efp_output_ds[corr_var_name].values

            sns.heatmap(corr_value_arr, xticklabels=vars_for_heatmap, yticklabels=vars_for_heatmap, vmin=0., vmax=1.0, cmap='viridis', annot=True, fmt='.2f')

            plt.savefig(f'{plot_dir_corr2}/TEM_term_{hemisphere}_corr_matrix.pdf')          
        plt.close('all')

def efp_annual_cycle(plot_dir, efp_output_ds, season_month_dict, use_500hPa_only):

    plot_dir_corr = f'{plot_dir}/efp_plots/'
    if use_500hPa_only:
        plot_dir_corr = plot_dir_corr + '500hPa_only/'    

    if not os.path.isdir(plot_dir_corr):
        os.makedirs(plot_dir_corr)

    season_list = [season_val for season_val in season_month_dict.keys()]


    central_month_dict = {'DJF':7,
                     'JFM':8,
                     'FMA':9,
                     'MAM':10,
                     'AMJ':11,
                     'MJJ':12,
                     'JJA':1, 
                     'JAS':2,
                     'ASO':3,
                     'SON':4,
                     'OND':5,
                     'NDJ':6,
                    }   

    season_list = [season_val for season_val in central_month_dict.keys()]

    for hemisphere in ['n', 's']:
        plot_dir_corr2 = f'{plot_dir_corr}/{hemisphere}_hemisphere/annual_cycle/'

        if not os.path.isdir(plot_dir_corr2):
            os.makedirs(plot_dir_corr2)       

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        efp_arr_dict = {}
        central_month_arr_dict = {}

        for var_to_analyse in ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']:
            efp_arr = np.zeros(len(season_list))
            signed_efp_arr = np.zeros(len(season_list))
            central_month_arr = np.zeros(len(season_list))
            central_month_labels = np.array(['' for val in range(len(season_list))])

            for time_frame in season_list:
                efp_var_name = f'efp_{var_to_analyse}_ucomp_{hemisphere}_{time_frame}'
                signed_efp_var_name = f'signed_efp_{var_to_analyse}_ucomp_{hemisphere}_{time_frame}'

                if efp_var_name in efp_output_ds.keys():
                    efp_val = efp_output_ds[efp_var_name]
                else:
                    efp_val = np.nan                

                if signed_efp_var_name in efp_output_ds.keys():
                    signed_efp_val = efp_output_ds[signed_efp_var_name]
                else:
                    signed_efp_val = np.nan     

                efp_arr[central_month_dict[time_frame]-1] = efp_val
                signed_efp_arr[central_month_dict[time_frame]-1] = signed_efp_val

                central_month_arr[central_month_dict[time_frame]-1] = central_month_dict[time_frame]
                central_month_labels[central_month_dict[time_frame]-1] = time_frame[1]

            efp_arr_dict[var_to_analyse] = efp_arr
            central_month_arr_dict[var_to_analyse] = central_month_arr

            ax1.plot(central_month_arr, signed_efp_arr, label=var_to_analyse, marker='x')
            ax2.plot(central_month_arr, efp_arr, label=var_to_analyse, marker='x')

        ax2.plot(central_month_arr, efp_arr_dict['div1_QG_123']+efp_arr_dict['div1_QG_gt3'], label='123 + gt3', marker='x')
        ax2.plot(central_month_arr, efp_arr_dict['div1_QG']-efp_arr_dict['div1_QG_123'], label='tot - 123', marker='x')


        # ax = plt.gca()

        for ax_to_use in [ax1, ax2]:
            ax_to_use.set_xticks(range(1, len(central_month_labels)+1))
            ax_to_use.set_xticklabels(central_month_labels)            
            ax_to_use.legend()

        # Move x-axis to y=0
        ax1.spines['bottom'].set_position(('data', 0))
        # Hide top and right spines for a cleaner look
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)     

        ax1.set_ylim((-1., 1.0))
        ax2.set_ylim((0., 1.0))        

        ax1.set_title('Signed efp annual cycle')
        ax2.set_title('Standard efp annual cycle')        

        plt.savefig(f'{plot_dir_corr2}/efp_annual_cycle.pdf')


def big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir, use_qg=False):

    plot_dir_TEM = f'{plot_dir}/TEM_plots/'

    if not os.path.isdir(plot_dir_TEM):
        os.makedirs(plot_dir_TEM)

    # Create the figure and 6 subplots
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)  # 2 rows, 4 columns

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    make_line_plots=False

    if use_qg:
        terms_in_budget  = ['delta_ubar_dt', 'fvbarstar', 'div1_QG', 'div2_QG',] 
        qg_str = '_QG'   
        if np.isnan(dataset['div2_QG'].max().values):
            make_line_plots=True

    else:
        terms_in_budget  = ['delta_ubar_dt', 'fvbarstar', 'vbarstar_1oacosphi_dudphi', 'omegabarstar_dudp', 'div1', 'div2',]
        qg_str = ''

    if include_udt_rdamp:
        terms_in_budget.append('udt_rdamp')

    if use_qg:    
        terms_in_budget.append('total_tend_QG')
    else:
        terms_in_budget.append('total_tend')        

    day_length = 86400.

    #EP flux divergence is in m/s/day, so convert other terms to m/s/day for comparison

    scaling_dict = {'fvbarstar':1*day_length,
                'vbarstar_1oacosphi_dudphi':-1*day_length,
                'omegabarstar_dudp': -1*day_length,
                'div1':1., 
                'div2':1., 
                'div1_QG':1., 
                'div2_QG':1.,                 
                'udt_rdamp':1.*day_length,
                'delta_ubar_dt':1.*day_length,
                'total_tend':1.*day_length,
                'total_tend_QG':1.*day_length                
                }

    levels = np.linspace(-18., 18., num=61, endpoint=True)

    # Loop to create each subplot
    for i, ax in enumerate(axes):
        if i < len(terms_in_budget):
            var_name_to_plot = terms_in_budget[i]

            data_to_plot = dataset[var_name_to_plot]*scaling_dict[var_name_to_plot]

            if 'time' in data_to_plot.dims:
                data_to_plot = data_to_plot.mean('time')

            if 'lon' in data_to_plot.dims:
                data_to_plot = data_to_plot.mean('lon')

            # logging.info(f'{data_to_plot.max().values} {data_to_plot.min().values} {np.where(np.isnan(data_to_plot.values))[0].shape}')
            if make_line_plots:
                contour = ax.plot(dataset['lat'], data_to_plot.sel(pfull=500.))
            else:
                contour = ax.contourf(dataset['lat'], dataset['pfull'], data_to_plot, cmap='RdBu_r', levels=levels)
                ax.set_ylim(1000., 0.)

            ax.set_title(plot_title_dict[var_name_to_plot])

    if not make_line_plots:
        cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label("m/s/day")

    if use_qg:
        if make_line_plots:
            plt.suptitle('Terms in QG TEM budget at 500hPa')               
        else:
            plt.suptitle('Terms in QG TEM budget')   
    else:
        plt.suptitle('Terms in TEM budget')   

    plt.savefig(f'{plot_dir_TEM}/TEM_terms{qg_str}.pdf')   
    plt.close('all')

def eof_plots(eof_vars, eof_ds, n_eofs, season_month_dict, lag_len, plot_dir, plot_title_dict, propogate_all_nans):

    plot_dir_EOF = f'{plot_dir}/EOF_plots/'

    if not os.path.isdir(plot_dir_EOF):
        os.makedirs(plot_dir_EOF)

    color_list = ['blue', 'orange', 'green']

    if propogate_all_nans:
        prop_nan_str = '_prop_nans'
    else:
        prop_nan_str=''

    eof_ds_coords = [coord for coord in eof_ds.coords.keys()]
    eof_ds_vars   = [var for var in eof_ds.variables.keys() if var not in eof_ds_coords]

    season_list = [season_val for season_val in season_month_dict.keys()]

    all_time_season_list = season_list+['all_time']

    va_str_dict = {True:'_va', False:'', 500.:'_500'}
# AUTOCORRELATION PLOTS
    do_autocorrelation_plots=False

    logging.info(f'doing autocorrelation plots')

    for hemisphere in ['n','s']:
        for time_frame in all_time_season_list:
            for use_va in [True, False, 500.]:
                va_str = va_str_dict[use_va]
                if do_autocorrelation_plots:                
                    for eof_var in eof_vars:
                        pc_orig_var_name = f'{eof_var}{va_str}_PCs_{hemisphere}_{time_frame}'
                        pc_ucomp_proj_name = f'{eof_var}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'
                        if pc_ucomp_proj_name in eof_ds_vars:
                            pc_var_name_list = [pc_orig_var_name, pc_ucomp_proj_name]
                        else:
                            pc_var_name_list = [pc_orig_var_name]

                        for pc_var_name in pc_var_name_list:
                            plt.figure()
                            for eof_num_idx in eof_ds['eof_num'].values:
                                x_corr = eof_ds[pc_var_name][eof_num_idx,:].values
                                y_corr = eof_ds[pc_var_name][eof_num_idx,:].values

                                # correlation = signal.correlate(x_corr, y_corr, mode="full")
                                # lags = signal.correlation_lags(x_corr.size, y_corr.size, mode="full")

                                # cross_corr2, lags2 = cross_correlation(x_corr, y_corr, max_lag=lag_len)

                                # eof_ds.coords['time'] = dataset['time']
                                ntime = eof_ds.coords['time'].shape[0]

                                if time_frame!='all_time':
                                    where_hem = np.where(eof_ds['time'].dt.month.isin(season_month_dict[time_frame])) 

                                    eof_pc_all_time = np.zeros((n_eofs,ntime))+np.nan
                                    eof_pc_all_time[:, where_hem[0]] =  eof_ds[pc_var_name].values
                                else:
                                    eof_pc_all_time =  eof_ds[pc_var_name].values

                                cross_corr_neg_nan, lags = eff.cross_correlation(eof_pc_all_time[eof_num_idx,:], eof_pc_all_time[eof_num_idx,:], lag_len)
                                cross_corr_neg_nan_verif, lags_verif = eff.cross_correlation(x_corr, y_corr, lag_len)

                                cross_corr3_neg = eff.sm_cross_correlation(x_corr, y_corr, lag_len) #y leads x
                                cross_corr3_pos = eff.sm_cross_correlation(y_corr, x_corr, lag_len) #x leads y  
                                pos_lags = np.arange(lag_len)     
                                neg_lags = np.arange(0,-lag_len, -1)     


                                # plt.plot(lags, correlation, label='scipy')
                                # plt.plot(lags2, cross_corr2, label=f'{eof_var} EOF {eof_num_idx}', marker='*')
                                plt.plot(pos_lags, cross_corr3_pos, label=f'{eof_var} EOF {eof_num_idx+1}', color=color_list[eof_num_idx])
                                plt.plot(neg_lags, cross_corr3_neg, color=color_list[eof_num_idx])        
                                # plt.plot(lags, cross_corr_neg_nan, label=f'{eof_var} EOF {eof_num_idx+1}', color=color_list[eof_num_idx], linestyle='--')
                                # plt.plot(lags_verif, cross_corr_neg_nan_verif, label=f'{eof_var} EOF {eof_num_idx+1}', color=color_list[eof_num_idx], linestyle='-.')  

                            plt.xlabel('lag (days)')
                            plt.ylabel('lagged correlation')
                            plt.xlim(-30.,30.)
                            plt.ylim(-1.0, 1.0)
                            ax=plt.gca()
                            # Move x-axis and y-axis to zero
                            ax.spines['left'].set_position('zero')
                            ax.spines['bottom'].set_position('zero')

                            # Hide top and right spines
                            ax.spines['right'].set_color('none')
                            ax.spines['top'].set_color('none')     
                            # Adjust ticks
                            ax.xaxis.set_ticks_position('bottom')
                            ax.yaxis.set_ticks_position('left')       

                            plt.title(f'{plot_title_dict[eof_var]} {hemisphere}H lagged autocorrelation')
                            plt.legend()

                            plot_dir_EOF_auto = f'{plot_dir_EOF}/autocorrelation/{hemisphere}_hemisphere/{time_frame}/{va_str}'

                            if not os.path.isdir(plot_dir_EOF_auto):
                                os.makedirs(plot_dir_EOF_auto)

                            plt.savefig(f'{plot_dir_EOF_auto}/{pc_var_name}_lagged_autocorrelation{prop_nan_str}.pdf')
                            plt.close()                    

                logging.info(f'doing crosscorrelation plots')                        
# CROSSCORRELATION PLOTS
                do_crosscorrelation_plots=False
                if do_crosscorrelation_plots:
                    for eof_var1 in eof_vars:
                        for eof_var2 in eof_vars:    
                            pc_orig_var1_name = f'{eof_var1}{va_str}_PCs_{hemisphere}_{time_frame}'
                            pc_orig_var2_name = f'{eof_var2}{va_str}_PCs_{hemisphere}_{time_frame}'

                            pc_ucomp_proj2_name = f'{eof_var2}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'
                            if pc_ucomp_proj2_name in eof_ds_vars:
                                pc_var2_name_list = [pc_orig_var2_name, pc_ucomp_proj2_name]
                            else:
                                pc_var2_name_list = [pc_orig_var2_name]

                            for pc_var2_name in pc_var2_name_list:

                                plt.figure()
                                for eof_num_idx in [0]:
                                    x_corr = eof_ds[pc_orig_var1_name][eof_num_idx,:].values
                                    y_corr = eof_ds[pc_var2_name][eof_num_idx,:].values


                                    if time_frame!='all_time':
                                        where_hem = np.where(eof_ds['time'].dt.month.isin(season_month_dict[time_frame])) 

                                        eof_pc_all_time1 = np.zeros((n_eofs,ntime))+np.nan
                                        eof_pc_all_time2 = np.zeros((n_eofs,ntime))+np.nan

                                        eof_pc_all_time1[:, where_hem[0]] =  eof_ds[pc_orig_var1_name].values
                                        eof_pc_all_time2[:, where_hem[0]] =  eof_ds[pc_var2_name].values
                                    else:
                                        eof_pc_all_time1 = eof_ds[pc_orig_var1_name].values
                                        eof_pc_all_time2 = eof_ds[pc_var2_name].values                        

                                    cross_corr_neg_nan, lags = eff.cross_correlation(eof_pc_all_time1[eof_num_idx,:], eof_pc_all_time2[eof_num_idx,:], lag_len)
                                    cross_corr_neg_nan_verif, lags_verif = eff.cross_correlation(x_corr, y_corr, lag_len)
                                    # cross_corr2, lags2 = cross_correlation(x_corr, y_corr, max_lag=300)

                                    cross_corr3_neg = eff.sm_cross_correlation(x_corr, y_corr, lag_len) #y leads x
                                    cross_corr3_pos = eff.sm_cross_correlation(y_corr, x_corr, lag_len) #x leads y  
                                    pos_lags = np.arange(lag_len)     
                                    neg_lags = np.arange(0,-lag_len, -1)    

                                    # plt.plot(lags, correlation, label='scipy')
                                    # plt.plot(lags2, cross_corr2, label=f'{eof_var} EOF {eof_num_idx}', marker='*')
                                    plt.plot(pos_lags, cross_corr3_pos, label=f'{eof_var1} {eof_var2}', color=color_list[eof_num_idx])
                                    plt.plot(neg_lags, cross_corr3_neg, color=color_list[eof_num_idx])    
                                    plt.plot(lags, cross_corr_neg_nan, label=f'{eof_var1} {eof_var2}', color=color_list[eof_num_idx], linestyle='--') 
                                    plt.plot(lags_verif, cross_corr_neg_nan_verif, label=f'{eof_var1} {eof_var2}', color=color_list[eof_num_idx], linestyle='-.') 

                                plt.xlabel('lag (days)')
                                plt.ylabel('lagged correlation')
                                plt.xlim(-30.,30.)
                                plt.ylim(-1.0, 1.0)
                                plt.title(f'{plot_title_dict[eof_var1]} {hemisphere}H lagged correlation with {plot_title_dict[eof_var2]} ')
                                ax=plt.gca()
                                # Move x-axis and y-axis to zero
                                ax.spines['left'].set_position('zero')
                                ax.spines['bottom'].set_position('zero')

                                # Hide top and right spines
                                ax.spines['right'].set_color('none')
                                ax.spines['top'].set_color('none')   
                                # Adjust ticks
                                ax.xaxis.set_ticks_position('bottom')
                                ax.yaxis.set_ticks_position('left')    
                                ax.text(15, 0.5, f'{plot_title_dict[eof_var1]} leads', fontsize=12, color='black', ha='center', va='center')
                                ax.text(-15, 0.5, f'{plot_title_dict[eof_var2]} leads', fontsize=12, color='black', ha='center', va='center')        

                                plt.legend()
                                plot_dir_EOF_cross = f'{plot_dir_EOF}/cross_correlation/{hemisphere}_hemisphere/{time_frame}/{eof_var1}/{va_str}'

                                if not os.path.isdir(plot_dir_EOF_cross):
                                    os.makedirs(plot_dir_EOF_cross)

                                plt.savefig(f'{plot_dir_EOF_cross}/{eof_var1}_{pc_var2_name}_lagged{prop_nan_str}.pdf')
                                plt.close()

# EOF lat-pressure PLOTS
            for eof_var in tqdm(eof_vars):

                logging.info(f'making EOF lat-pres plot for {eof_var}')
                # Create the figure and 6 subplots
                fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)  # 2 rows, 4 columns

                # Flatten the axes array for easy iteration
                axes = axes.flatten()

                level_max = np.nanmax(np.abs(eof_ds[f'{eof_var}_EOFs_{hemisphere}_{time_frame}'].values))

                levels = np.linspace(-level_max, level_max, num=31, endpoint=True)

                if hemisphere=='n':
                    lat_sign = 1.
                else:
                    lat_sign = -1.

                for i, ax in enumerate(axes):
                    contour = ax.contourf(lat_sign*eof_ds.lat.values, eof_ds.pfull.values, eof_ds[f'{eof_var}_EOFs_{hemisphere}_{time_frame}'][i, ...], cmap='RdBu_r', levels=levels)
                    line_contours = ax.contour(lat_sign*eof_ds.lat.values, eof_ds.pfull.values, eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'],levels=11, colors='grey')
                    zero_contours = ax.contour(lat_sign*eof_ds.lat.values, eof_ds.pfull.values, eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'],levels=[0.], colors='grey', linestyles='dashed')                    
                    ax.clabel(line_contours, line_contours.levels, fontsize=10)
                    ax.clabel(zero_contours, zero_contours.levels, fontsize=10)
                    ax.set_title(f"EOF {i+1} explains {eof_ds[f'{eof_var}_var_frac_{hemisphere}_{time_frame}'].values[i]*100.:4.1f}%")
                    ax.set_ylim(1000., 0.)

                cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
                # cbar.set_label("m/s/day")

                plot_dir_EOF_EOFs = f'{plot_dir_EOF}/EOFs/{hemisphere}_hemisphere/{time_frame}/'

                if not os.path.isdir(plot_dir_EOF_EOFs):
                    os.makedirs(plot_dir_EOF_EOFs)

                plt.savefig(f'{plot_dir_EOF_EOFs}/{eof_var}_{hemisphere}_{time_frame}_EOFs{prop_nan_str}.pdf')
                plt.close()

#EOF vert_av plots
            for eof_var in tqdm(eof_vars):

                logging.info(f'making EOF va plot for {eof_var}')
                # Create the figure and 6 subplots
                fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)  # 2 rows, 4 columns

                # Flatten the axes array for easy iteration
                axes = axes.flatten()

                level_max = np.nanmax(np.abs(eof_ds[f'{eof_var}_va_EOFs_{hemisphere}_{time_frame}']))

                if hemisphere=='n':
                    lat_sign = 1.
                else:
                    lat_sign = -1.

                for i, ax in enumerate(axes):
                    line_plot = ax.plot(lat_sign*eof_ds.lat.values, eof_ds[f'{eof_var}_va_EOFs_{hemisphere}_{time_frame}'][i, ...])
                    plt.ylim((-level_max, level_max))
                    ax.set_title(f"EOF {i+1} explains {eof_ds[f'{eof_var}_va_var_frac_{hemisphere}_{time_frame}'].values[i]*100.:4.1f}%")


                plot_dir_EOF_EOFs = f'{plot_dir_EOF}/EOFs/{hemisphere}_hemisphere/{time_frame}/va/'

                if not os.path.isdir(plot_dir_EOF_EOFs):
                    os.makedirs(plot_dir_EOF_EOFs)

                plt.savefig(f'{plot_dir_EOF_EOFs}/{eof_var}_va_{hemisphere}_{time_frame}_EOFs{prop_nan_str}.pdf')
                plt.close()

# Time-mean PLOTS
            for eof_var in tqdm(eof_vars):

                logging.info(f'making time-mean plot for {eof_var}')
                # Create the figure and 6 subplots
                fig, axes = plt.subplots(1, 1, figsize=(12, 6), sharex=True, sharey=True)  # 2 rows, 4 columns

                # Flatten the axes array for easy iteration
                # axes = axes.flatten()

                level_max = np.nanmax(eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'])
                level_min = np.nanmin(eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'])

                if np.sign(level_max)==np.sign(level_min):
                    levels = np.linspace(level_min, level_max, num=30, endpoint=True)            
                else:
                    level_max = np.max([np.abs(level_min), np.abs(level_max)])
                    levels = np.linspace(-level_max, level_max, num=31, endpoint=True)

                # for i, ax in enumerate(axes):
                ax = plt.gca()
                contour = ax.contourf(eof_ds.lat.values, eof_ds.pfull.values, eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'], cmap='RdBu_r', levels=levels)
                ax.set_title(f'{eof_var} in {hemisphere}')
                ax.set_ylim(1000., 0.)

                cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
                # cbar.set_label("m/s/day")

                plot_dir_EOF_TM = f'{plot_dir_EOF}/time_mean_fields/{hemisphere}_hemisphere/{time_frame}/'

                if not os.path.isdir(plot_dir_EOF_TM):
                    os.makedirs(plot_dir_EOF_TM)

                plt.savefig(f'{plot_dir_EOF_TM}/{eof_var}_{hemisphere}_{time_frame}_mean{prop_nan_str}.pdf')   
                plt.close()    

    plt.close('all')

# Projection comparison PLOTS
    for hemisphere in ['n','s']:
        for time_frame in all_time_season_list:
            for eof_var in ['ucomp', 'div1_QG']:
                plt.figure()
                proj_data = eof_ds[f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}'][0, :].values
                orig_data = eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'][0,:].values
                plt.plot(orig_data, proj_data, linestyle='none', marker='x')
                plt.xlabel(f'{eof_var}_PCs_{hemisphere}_{time_frame}')
                plt.ylabel(f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}')
                plt.title('EOF1 original PC vs PC projected onto ucomp EOF1')

                plot_dir_EOF_proj = f'{plot_dir_EOF}/proj_comparisons/{hemisphere}_hemisphere/{time_frame}/'

                if not os.path.isdir(plot_dir_EOF_proj):
                    os.makedirs(plot_dir_EOF_proj)

                plt.savefig(f'{plot_dir_EOF_proj}/{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}_vs_orig_PC{prop_nan_str}.pdf')       


def plot_power_spectrum(power_spec_ds, ps_var, va_str, plot_dir_PS, scaling, time_name):
    plt.figure()
    # plt.plot(power_spec_ds['frequency'], power_spec_ds[f'{ps_var}_power_spec_welch'], label='welch')
    plt.plot(power_spec_ds[f'frequency_{time_name}'], 2.*power_spec_ds[f'{ps_var}_power_spec_stft'], linestyle='--', label='stft') #for some reason there's a factor of 2 difference between these. Presumably in the normalisation somewhere?
    plt.legend()
    plt.xlim(0., 0.25)
    plt.title(f'Power spectrum of {ps_var}{va_str} using STFT method')
    plt.xlabel('frequency (1/days)')
    plt.savefig(f'{plot_dir_PS}/{ps_var}{va_str}_PC1_power_spectrum_stft_{scaling}.pdf')                    

def plot_multiple_power_spectra(var_list, power_spec_ds, ps_var, va_str, plot_dir_PS, scaling, use_div1_proj, hemisphere, time_frame, time_name):
    plt.figure()

    plot_name_str = ''

    for ps_var in var_list:
        if use_div1_proj:
            div1_name = f'{ps_var}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'
        else:
            div1_name = f'{ps_var}{va_str}_PCs_{hemisphere}_{time_frame}'

        plt.plot(power_spec_ds[f'frequency_{time_name}'], 2.*power_spec_ds[f'{div1_name}_power_spec_stft'], label=f'{div1_name}{va_str}') #for some reason there's a factor of 2 difference between these. Presumably in the normalisation somewhere?
        plot_name_str = plot_name_str+f'{ps_var}{va_str}_'

    plt.legend()
    plt.xlim(0., 0.25)
    plt.title(f'Power spectrum of {plot_name_str} using STFT method')
    plt.xlabel('frequency (1/days)')
    plt.savefig(f'{plot_dir_PS}/{plot_name_str}PC1_power_spectrum_stft_{scaling}.pdf')    


def plot_coherence_cospectrum_phase_diff(power_spec_ds, ucomp_name, div1_name, plot_dir_PS, va_str, scaling, time_name):

    frequency_name = f'frequency_{time_name}'
    plt.figure()
    # plt.plot(power_spec_ds[frequency_name], np.real(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_welch']), label='real Welch')
    # plt.plot(power_spec_ds[frequency_name], np.imag(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_welch']), label='Imag Welch')
    plt.plot(power_spec_ds[frequency_name], np.real(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft']), label='real STFT', linestyle='--')
    plt.plot(power_spec_ds[frequency_name], np.imag(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft']), label='Imag STFT', linestyle='--')    
    plt.plot(power_spec_ds[frequency_name], 2.*np.pi*power_spec_ds[frequency_name], label='2piomega')
    plt.xlim(0., 0.25)
    plt.legend()
    plt.title(f'Cospectrum of ucomp and div1')
    plt.xlabel('frequency (1/days)')
    plt.savefig(f'{plot_dir_PS}/ucomp{va_str}_{div1_name}{va_str}_PC1_cospectrum_{scaling}.pdf')    

    plt.figure()
    # plt.plot(power_spec_ds[frequency_name], power_spec_ds[f'{ucomp_name}_{div1_name}_coher_welch']**2., label='welch')
    plt.plot(power_spec_ds[frequency_name], power_spec_ds[f'{ucomp_name}_{div1_name}_coher_stft']**2., label='stft', linestyle='--')
    plt.xlim(0., 0.25)
    plt.legend()
    plt.title(f'Coherence squared of ucomp and div1 using stft method')
    plt.xlabel('frequency (1/days)')
    plt.savefig(f'{plot_dir_PS}/ucomp{va_str}_{div1_name}{va_str}_PC1_coher_sqd.pdf')      

    plt.figure()
    plt.plot(power_spec_ds[frequency_name], power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff'], label='data')
    plt.plot(power_spec_ds[frequency_name], np.rad2deg(np.arctan(2.*np.pi*power_spec_ds[frequency_name]*power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_1'])), linestyle='--', label=f"fit with {power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_1']:4.2f} days")
    plt.plot(power_spec_ds[frequency_name], np.rad2deg(np.arctan(2.*np.pi*power_spec_ds[frequency_name]*power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_2'])), linestyle='--', label=f"fit with {power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_2']:4.2f} days")
    plt.plot(power_spec_ds[frequency_name], np.rad2deg(np.arctan(2.*np.pi*power_spec_ds[frequency_name]*power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'])), linestyle='--', label=f"fit with {power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3']:4.2f} days")
    plt.xlim(0., 0.25)
    plt.legend()
    plt.title(f'Phase of ucomp and div1 using stft method')
    plt.xlabel('frequency (1/days)')
    plt.savefig(f'{plot_dir_PS}/ucomp{va_str}_{div1_name}{va_str}_PC1_phase_diff.pdf')        

def plot_b_annual_cycle(b_dataset, season_month_dict, plot_dir):

    va_str_dict = {True:'_va', False:'', 500.:'_500'}


    central_month_dict = {'DJF':7,
                     'JFM':8,
                     'FMA':9,
                     'MAM':10,
                     'AMJ':11,
                     'MJJ':12,
                     'JJA':1, 
                     'JAS':2,
                     'ASO':3,
                     'SON':4,
                     'OND':5,
                     'NDJ':6,
                    }   

    season_list = [season_val for season_val in central_month_dict.keys()]

    for hemisphere in ['n', 's']:#, 's_DJF']:
        for use_va in [True, False, 500.]:
            va_str = va_str_dict[use_va]   
            plot_dir_b = f'{plot_dir}/b_plots/{hemisphere}_hemisphere/annual_cycle/{va_str}'

            if not os.path.isdir(plot_dir_b):
                os.makedirs(plot_dir_b)

            plt.figure()
            for var_to_analyse in ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']:
                b_arr = np.zeros(len(season_list))
                central_month_arr = np.zeros(len(season_list))
                central_month_labels = np.array(['' for val in range(len(season_list))])

                for time_frame in season_list:
                    b_var_name = f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'
                    if b_var_name in b_dataset.keys():
                        b_av = b_dataset[b_var_name].mean('lag').values    
                    else:
                        b_av = np.nan

                    b_arr[central_month_dict[time_frame]-1] = b_av
                    central_month_arr[central_month_dict[time_frame]-1] = central_month_dict[time_frame]
                    central_month_labels[central_month_dict[time_frame]-1] = time_frame[1]

                plt.plot(central_month_arr, b_arr, label=var_to_analyse, marker='x')
            ax = plt.gca()

            ax.set_xticks(range(1, len(central_month_labels)+1))
            ax.set_xticklabels(central_month_labels)            
            # Move x-axis to y=0
            ax.spines['bottom'].set_position(('data', 0))
            # Hide top and right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)     
            plt.ylim((-0.1, 0.1))
            plt.legend()
            plt.savefig(f'{plot_dir_b}/ucomp{va_str}_{var_to_analyse}_b_annual_cycle.pdf')