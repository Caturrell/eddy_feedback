import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np
import eddy_feedback_functions as eff
from tqdm import tqdm
import os

def individual_plots(dataset, individual_plot_list, plot_dir):
    """
    Generate individual contour plots for each variable in the provided list.

    For each variable, the function computes a time (and longitude) mean if present,
    creates a contour plot of the variable using latitude and pressure coordinates,
    and saves the plot as a PDF in a designated directory.

    Parameters:
        dataset (xarray.Dataset): Dataset containing the variables to plot.
        individual_plot_list (list): List of variable names (strings) to plot.
        plot_dir (str): Root directory to save the plots.
    
    Returns:
        None
    """
    # Create a subdirectory for individual plots if it doesn't exist.
    plot_dir_indiv = f'{plot_dir}/individual_plots/'
    if not os.path.isdir(plot_dir_indiv):
        os.makedirs(plot_dir_indiv)

    # Loop over each variable name provided.
    for var_to_plot in individual_plot_list:
        logging.info(f'plotting {var_to_plot}')
        data_to_plot = dataset[var_to_plot]

        # If the data has a time dimension, average over time.
        if 'time' in data_to_plot.dims:
            data_to_plot = data_to_plot.mean('time')
        # If the data has a longitude dimension, average over longitude.
        if 'lon' in data_to_plot.dims:
            data_to_plot = data_to_plot.mean('lon')        

        # Create a new figure and plot a contour using lat and pfull coordinates.
        plt.figure()
        plt.contourf(dataset['lat'], dataset['pfull'], data_to_plot, cmap='RdBu_r', levels=31)
        plt.colorbar()
        plt.ylim(1000., 0.)
        plt.savefig(f'{plot_dir_indiv}/{var_to_plot}.pdf')

    plt.close('all')


def individual_corr_plots(efp_output_ds, dataset, vars_to_correlate, plot_dir):
    """
    Generate correlation contour plots for selected variables.

    For each variable in vars_to_correlate, the function constructs a plot showing
    the correlation (averaged over time and longitude) on latitude-pressure space.
    The plot title includes the eddy-feedback parameter (efp) value.

    Parameters:
        efp_output_ds (xarray.Dataset): Dataset containing EFP outputs and correlation fields.
        dataset (xarray.Dataset): Original dataset for coordinate references.
        vars_to_correlate (list): List of base variable names to correlate.
        plot_dir (str): Directory where the correlation plots will be saved.
    
    Returns:
        None
    """
    # Create a directory for EFP plots if it does not exist.
    plot_dir_corr = f'{plot_dir}/efp_plots/'
    if not os.path.isdir(plot_dir_corr):
        os.makedirs(plot_dir_corr)

    # Loop over hemispheres.
    for hemisphere in ['n', 's']:
        plot_dir_corr2 = f'{plot_dir_corr}/{hemisphere}_hemisphere/'
        if not os.path.isdir(plot_dir_corr2):
            os.makedirs(plot_dir_corr2)

        # For each variable, compute the variable name to plot (assumed to include 'ucomp' and hemisphere info).
        for var_to_plot_tick in vars_to_correlate:
            var_to_plot = f'{var_to_plot_tick}_ucomp_{hemisphere}_corr'
            logging.info(f'plotting {var_to_plot}')
            data_to_plot = efp_output_ds[var_to_plot]

            # Average over time if available.
            if 'time' in data_to_plot.dims:
                data_to_plot = data_to_plot.mean('time')
            # Average over longitude if available.
            if 'lon' in data_to_plot.dims:
                data_to_plot = data_to_plot.mean('lon')

            # Define contour levels between 0 and 1.
            levels = np.linspace(0., 1., num=40, endpoint=True)

            # Create a new figure and plot a filled contour.
            plt.figure()
            plt.contourf(dataset['lat'], dataset['pfull'], data_to_plot, cmap='viridis', levels=levels)
            plt.colorbar()
            plt.ylim(1000., 0.)
            # Title incorporates the eddy-feedback parameter value from efp_output_ds.
            plt.title(f"efp for {var_to_plot_tick} is {efp_output_ds[f'efp_{var_to_plot_tick}_ucomp_{hemisphere}'].values}")
            plt.savefig(f'{plot_dir_corr2}/{var_to_plot}_{hemisphere}.pdf')

    plt.close('all')


def heatmap_tem_plot(vars_to_correlate, plot_dir, efp_output_ds):
    """
    Create a heatmap of correlation values for selected TEM terms.

    Constructs a heatmap showing the correlations between variables
    (excluding 'udt_rdamp' and including 'ucomp').

    Parameters:
        vars_to_correlate (list): List of variable names (strings) for which to plot correlations.
        plot_dir (str): Directory where the heatmap plot will be saved.
        efp_output_ds (xarray.Dataset): Dataset containing the EFP and correlation data.
    
    Returns:
        None
    """
    # Create a directory for EFP plots if it does not exist.
    plot_dir_corr = f'{plot_dir}/efp_plots/'
    if not os.path.isdir(plot_dir_corr):
        os.makedirs(plot_dir_corr)

    # Exclude 'udt_rdamp' and add 'ucomp' to the list for the heatmap.
    vars_for_heatmap = [var for var in vars_to_correlate if var != 'udt_rdamp'] + ['ucomp']

    # Loop over hemispheres.
    for hemisphere in ['n', 's']:
        plt.figure(figsize=(16, 16))
        n_corr_vars = len(vars_for_heatmap)

        # Initialize an array to store correlation values.
        corr_value_arr = np.zeros((n_corr_vars, n_corr_vars))

        # Fill the array by looking up corresponding EFP values.
        for idx_2, var2_to_correlate in enumerate(vars_for_heatmap):
            for idx_1, var_to_correlate in enumerate(vars_for_heatmap):
                corr_var_name = f'efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}'
                corr_value_arr[idx_2, idx_1] = efp_output_ds[corr_var_name].values

        # Use seaborn to plot the heatmap.
        sns.heatmap(corr_value_arr, xticklabels=vars_for_heatmap, yticklabels=vars_for_heatmap,
                    vmin=0., vmax=1.0, cmap='viridis', annot=True, fmt='.2f')
        plt.savefig(f'{plot_dir_corr}/TEM_term_{hemisphere}_corr_matrix.pdf')
    plt.close('all')


def big_TEM_plot(dataset, plot_title_dict, include_udt_rdamp, plot_dir, use_qg=False):
    """
    Create a composite plot of TEM budget terms in a grid layout.

    The function plots several TEM budget terms in subplots, scales them appropriately,
    and adds a common colorbar. The terms plotted depend on whether the QG approximation is used
    and if additional damping terms (udt_rdamp) should be included.

    Parameters:
        dataset (xarray.Dataset): Dataset containing TEM budget terms.
        plot_title_dict (dict): Dictionary mapping variable names to plot titles.
        include_udt_rdamp (bool): Flag indicating whether to include the 'udt_rdamp' term.
        plot_dir (str): Directory where the TEM plots will be saved.
        use_qg (bool): Flag to determine if QG (Quasi-Geostrophic) terms should be used (default: False).

    Returns:
        None
    """
    # Create a directory for TEM plots if it does not exist.
    plot_dir_TEM = f'{plot_dir}/TEM_plots/'
    if not os.path.isdir(plot_dir_TEM):
        os.makedirs(plot_dir_TEM)

    # Create a figure with 8 subplots (2 rows x 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    # Select the list of terms to plot based on whether QG terms are used.
    if use_qg:
        terms_in_budget = ['delta_ubar_dt', 'fvbarstar', 'div1_QG', 'div2_QG']
        qg_str = '_QG'
    else:
        terms_in_budget = ['delta_ubar_dt', 'fvbarstar', 'vbarstar_1oacosphi_dudphi', 'omegabarstar_dudp', 'div1', 'div2']
        qg_str = ''

    if include_udt_rdamp:
        terms_in_budget.append('udt_rdamp')

    if use_qg:
        terms_in_budget.append('total_tend_QG')
    else:
        terms_in_budget.append('total_tend')

    day_length = 86400.  # seconds per day

    # Define scaling factors to convert terms to m/s/day where necessary.
    scaling_dict = {'fvbarstar': 1 * day_length,
                    'vbarstar_1oacosphi_dudphi': -1 * day_length,
                    'omegabarstar_dudp': -1 * day_length,
                    'div1': 1., 
                    'div2': 1., 
                    'div1_QG': 1., 
                    'div2_QG': 1.,                 
                    'udt_rdamp': 1. * day_length,
                    'delta_ubar_dt': 1. * day_length,
                    'total_tend': 1. * day_length,
                    'total_tend_QG': 1. * day_length}

    levels = np.linspace(-18., 18., num=61, endpoint=True)

    # Loop over subplots and plot each TEM term.
    for i, ax in enumerate(axes):
        if i < len(terms_in_budget):
            var_name_to_plot = terms_in_budget[i]
            data_to_plot = dataset[var_name_to_plot] * scaling_dict[var_name_to_plot]

            # Average over time if present.
            if 'time' in data_to_plot.dims:
                data_to_plot = data_to_plot.mean('time')
            # Average over longitude if present.
            if 'lon' in data_to_plot.dims:
                data_to_plot = data_to_plot.mean('lon')

            # Create a filled contour plot.
            contour = ax.contourf(dataset['lat'], dataset['pfull'], data_to_plot, cmap='RdBu_r', levels=levels)
            ax.set_title(plot_title_dict[var_name_to_plot])
            ax.set_ylim(1000., 0.)
    # Add a common colorbar.
    cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("m/s/day")

    # Set overall title based on whether QG is used.
    if use_qg:
        plt.suptitle('Terms in QG TEM budget')
    else:
        plt.suptitle('Terms in TEM budget')

    plt.savefig(f'{plot_dir_TEM}/TEM_terms{qg_str}.pdf')
    plt.close('all')


def eof_plots(eof_vars, eof_ds, n_eofs, hemisphere_month_dict, lag_len, plot_dir, plot_title_dict, propogate_all_nans):
    """
    Generate multiple diagnostic plots related to EOF analysis.

    The function creates plots for lagged autocorrelations, cross-correlations between EOF PCs,
    spatial patterns of EOFs, time-mean fields, and comparisons of projected vs. original PCs.
    It organizes plots by hemisphere and seasonal/all-time grouping.

    Parameters:
        eof_vars (list): List of EOF variable names.
        eof_ds (xarray.Dataset): Dataset containing EOF results.
        n_eofs (int): Number of EOFs.
        hemisphere_month_dict (dict): Dictionary mapping hemispheres to relevant months.
        lag_len (int): Maximum lag (in days) for cross-correlation computations.
        plot_dir (str): Directory where all EOF-related plots will be saved.
        plot_title_dict (dict): Dictionary mapping variable names to descriptive titles.
        propogate_all_nans (bool): Flag to indicate if missing data have been propagated.
    
    Returns:
        None
    """
    # Create a directory for EOF plots if it does not exist.
    plot_dir_EOF = f'{plot_dir}/EOF_plots/'
    if not os.path.isdir(plot_dir_EOF):
        os.makedirs(plot_dir_EOF)

    # Define a list of colors for plotting different EOFs.
    color_list = ['blue', 'orange', 'green']

    # Determine suffix based on whether NaN propagation was used.
    prop_nan_str = '_prop_nans' if propogate_all_nans else ''

    # Determine variable names in the dataset.
    eof_ds_coords = [coord for coord in eof_ds.coords.keys()]
    eof_ds_vars = [var for var in eof_ds.variables.keys() if var not in eof_ds_coords]

    # Loop over hemispheres, time frames, and EOF variables to plot lagged autocorrelation.
    for hemisphere in ['n', 's']:
        for time_frame in ['season', 'all_time']:
            for eof_var in eof_vars:
                pc_orig_var_name = f'{eof_var}_PCs_{hemisphere}_{time_frame}'
                pc_ucomp_proj_name = f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}'
                if pc_ucomp_proj_name in eof_ds_vars:
                    pc_var_name_list = [pc_orig_var_name, pc_ucomp_proj_name]
                else:
                    pc_var_name_list = [pc_orig_var_name]

                for pc_var_name in pc_var_name_list:
                    plt.figure()
                    # Loop over each EOF index.
                    for eof_num_idx in eof_ds['eof_num'].values:
                        x_corr = eof_ds[pc_var_name][eof_num_idx, :].values
                        y_corr = eof_ds[pc_var_name][eof_num_idx, :].values

                        # For seasonal data, construct an array that fills with NaNs outside of season.
                        ntime = eof_ds.coords['time'].shape[0]
                        if time_frame == 'season':
                            where_hem = np.where(eof_ds['time'].dt.month.isin(hemisphere_month_dict[hemisphere]))
                            eof_pc_all_time = np.zeros((n_eofs, ntime)) + np.nan
                            eof_pc_all_time[:, where_hem[0]] = eof_ds[pc_var_name].values
                        else:
                            eof_pc_all_time = eof_ds[pc_var_name].values

                        # Compute cross-correlation using functions from eddy_feedback_functions.
                        cross_corr_neg_nan, lags = eff.cross_correlation(eof_pc_all_time[eof_num_idx, :], eof_pc_all_time[eof_num_idx, :], lag_len)
                        cross_corr_neg_nan_verif, lags_verif = eff.cross_correlation(x_corr, y_corr, lag_len)

                        cross_corr3_neg = eff.sm_cross_correlation(x_corr, y_corr, lag_len)  # y leads x
                        cross_corr3_pos = eff.sm_cross_correlation(y_corr, x_corr, lag_len)  # x leads y  
                        pos_lags = np.arange(lag_len)
                        neg_lags = np.arange(0, -lag_len, -1)

                        # Plot cross-correlation curves with different linestyles.
                        plt.plot(pos_lags, cross_corr3_pos, label=f'{eof_var} EOF {eof_num_idx}', color=color_list[eof_num_idx])
                        plt.plot(neg_lags, cross_corr3_neg, color=color_list[eof_num_idx])
                        plt.plot(lags, cross_corr_neg_nan, label=f'{eof_var} EOF {eof_num_idx+1}', color=color_list[eof_num_idx], linestyle='--')
                        plt.plot(lags_verif, cross_corr_neg_nan_verif, label=f'{eof_var} EOF {eof_num_idx+1}', color=color_list[eof_num_idx], linestyle='-.')
                    plt.xlabel('lag (days)')
                    plt.ylabel('lagged correlation')
                    plt.xlim(-30., 30.)
                    plt.ylim(-1.0, 1.0)
                    ax = plt.gca()
                    # Move the axes to cross at zero.
                    ax.spines['left'].set_position('zero')
                    ax.spines['bottom'].set_position('zero')
                    # Hide top and right spines.
                    ax.spines['right'].set_color('none')
                    ax.spines['top'].set_color('none')
                    # Adjust tick positions.
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    plt.title(f'{plot_title_dict[eof_var]} {hemisphere}H lagged autocorrelation')
                    plt.legend()

                    plot_dir_EOF_auto = f'{plot_dir_EOF}/autocorrelation/{hemisphere}_hemisphere/{time_frame}/'
                    if not os.path.isdir(plot_dir_EOF_auto):
                        os.makedirs(plot_dir_EOF_auto)
                    plt.savefig(f'{plot_dir_EOF_auto}/{pc_var_name}_lagged_autocorrelation{prop_nan_str}.pdf')

            # Loop for cross-correlation plots between different EOF variables.
            for eof_var1 in eof_vars:
                for eof_var2 in eof_vars:
                    pc_orig_var1_name = f'{eof_var1}_PCs_{hemisphere}_{time_frame}'
                    pc_orig_var2_name = f'{eof_var2}_PCs_{hemisphere}_{time_frame}'

                    pc_ucomp_proj2_name = f'{eof_var2}_PCs_from_ucomp_{hemisphere}_{time_frame}'
                    if pc_ucomp_proj2_name in eof_ds_vars:
                        pc_var2_name_list = [pc_orig_var2_name, pc_ucomp_proj2_name]
                    else:
                        pc_var2_name_list = [pc_orig_var2_name]

                    for pc_var2_name in pc_var2_name_list:
                        plt.figure()
                        for eof_num_idx in [0]:
                            x_corr = eof_ds[pc_orig_var1_name][eof_num_idx, :].values
                            y_corr = eof_ds[pc_var2_name][eof_num_idx, :].values

                            if time_frame == 'season':
                                where_hem = np.where(eof_ds['time'].dt.month.isin(hemisphere_month_dict[hemisphere]))
                                eof_pc_all_time1 = np.zeros((n_eofs, ntime)) + np.nan
                                eof_pc_all_time2 = np.zeros((n_eofs, ntime)) + np.nan
                                eof_pc_all_time1[:, where_hem[0]] = eof_ds[pc_orig_var1_name].values
                                eof_pc_all_time2[:, where_hem[0]] = eof_ds[pc_var2_name].values
                            else:
                                eof_pc_all_time1 = eof_ds[pc_orig_var1_name].values
                                eof_pc_all_time2 = eof_ds[pc_var2_name].values

                            cross_corr_neg_nan, lags = eff.cross_correlation(eof_pc_all_time1[eof_num_idx, :], eof_pc_all_time2[eof_num_idx, :], lag_len)
                            cross_corr_neg_nan_verif, lags_verif = eff.cross_correlation(x_corr, y_corr, lag_len)

                            cross_corr3_neg = eff.sm_cross_correlation(x_corr, y_corr, lag_len)  # y leads x
                            cross_corr3_pos = eff.sm_cross_correlation(y_corr, x_corr, lag_len)  # x leads y  
                            pos_lags = np.arange(lag_len)
                            neg_lags = np.arange(0, -lag_len, -1)

                            plt.plot(pos_lags, cross_corr3_pos, label=f'{eof_var1} {eof_var2}', color=color_list[eof_num_idx])
                            plt.plot(neg_lags, cross_corr3_neg, color=color_list[eof_num_idx])
                            plt.plot(lags, cross_corr_neg_nan, label=f'{eof_var1} {eof_var2}', color=color_list[eof_num_idx], linestyle='--')
                            plt.plot(lags_verif, cross_corr_neg_nan_verif, label=f'{eof_var1} {eof_var2}', color=color_list[eof_num_idx], linestyle='-.')
                        plt.xlabel('lag (days)')
                        plt.ylabel('lagged correlation')
                        plt.xlim(-30., 30.)
                        plt.ylim(-1.0, 1.0)
                        plt.title(f'{plot_title_dict[eof_var1]} {hemisphere}H lagged correlation with {plot_title_dict[eof_var2]}')
                        ax = plt.gca()
                        ax.spines['left'].set_position('zero')
                        ax.spines['bottom'].set_position('zero')
                        ax.spines['right'].set_color('none')
                        ax.spines['top'].set_color('none')
                        ax.xaxis.set_ticks_position('bottom')
                        ax.yaxis.set_ticks_position('left')
                        ax.text(15, 0.5, f'{plot_title_dict[eof_var1]} leads', fontsize=12, color='black', ha='center', va='center')
                        ax.text(-15, 0.5, f'{plot_title_dict[eof_var2]} leads', fontsize=12, color='black', ha='center', va='center')
                        plt.legend()
                        plot_dir_EOF_cross = f'{plot_dir_EOF}/cross_correlation/{hemisphere}_hemisphere/{time_frame}/{eof_var1}/'
                        if not os.path.isdir(plot_dir_EOF_cross):
                            os.makedirs(plot_dir_EOF_cross)
                        plt.savefig(f'{plot_dir_EOF_cross}/{eof_var1}_{pc_var2_name}_lagged{prop_nan_str}.pdf')

            # Plot EOF spatial patterns (EOFs) and associated variance fractions.
            for eof_var in tqdm(eof_vars):
                logging.info(f'making EOF plot for {eof_var}')
                fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)
                axes = axes.flatten()
                level_max = np.nanmax(np.abs(eof_ds[f'{eof_var}_EOFs_{hemisphere}_{time_frame}'].values))
                levels = np.linspace(-level_max, level_max, num=31, endpoint=True)
                for i, ax in enumerate(axes):
                    eof_key = f"{eof_var}_EOFs_{hemisphere}_{time_frame}"
                    var_frac_key = f"{eof_var}_var_frac_{hemisphere}_{time_frame}"
                    contour = ax.contourf(eof_ds.lat.values, eof_ds.pfull.values, eof_ds[eof_key][i, ...], cmap='RdBu_r', levels=levels)
                    ax.set_title(f"EOF {i+1} explains {eof_ds[var_frac_key].values[i] * 100.:4.1f}%")
                    ax.set_ylim(1000., 0.)
                cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
                plot_dir_EOF_EOFs = f'{plot_dir_EOF}/EOFs/{hemisphere}_hemisphere/{time_frame}/'
                if not os.path.isdir(plot_dir_EOF_EOFs):
                    os.makedirs(plot_dir_EOF_EOFs)
                plt.savefig(f'{plot_dir_EOF_EOFs}/{eof_var}_{hemisphere}_{time_frame}_EOFs{prop_nan_str}.pdf')

            # Plot time-mean fields of the EOF variables.
            for eof_var in tqdm(eof_vars):
                logging.info(f'making time-mean plot for {eof_var}')
                fig, axes = plt.subplots(1, 1, figsize=(12, 6), sharex=True, sharey=True)
                level_max = np.nanmax(eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'].values)
                level_min = np.nanmin(eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'].values)
                if np.sign(level_max) == np.sign(level_min):
                    levels = np.linspace(level_min, level_max, num=30, endpoint=True)
                else:
                    level_max = np.max([np.abs(level_min), np.abs(level_max)])
                    levels = np.linspace(-level_max, level_max, num=31, endpoint=True)
                ax = plt.gca()
                contour = ax.contourf(eof_ds.lat.values, eof_ds.pfull.values, eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'], cmap='RdBu_r', levels=levels)
                ax.set_title(f'{eof_var} in {hemisphere}')
                ax.set_ylim(1000., 0.)
                cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
                plot_dir_EOF_TM = f'{plot_dir_EOF}/time_mean_fields/{hemisphere}_hemisphere/{time_frame}/'
                if not os.path.isdir(plot_dir_EOF_TM):
                    os.makedirs(plot_dir_EOF_TM)
                plt.savefig(f'{plot_dir_EOF_TM}/{eof_var}_{hemisphere}_{time_frame}_mean{prop_nan_str}.pdf')
    plt.close('all')

    # Plot comparisons between original and projected principal components.
    for hemisphere in ['n', 's']:
        for time_frame in ['season', 'all_time']:
            for eof_var in ['ucomp', 'div1_QG']:
                plt.figure()
                proj_data = eof_ds[f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}'][0, :].values
                orig_data = eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'][0, :].values
                plt.plot(orig_data, proj_data, linestyle='none', marker='x')
                plt.xlabel(f'{eof_var}_PCs_{hemisphere}_{time_frame}')
                plt.ylabel(f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}')
                plt.title('EOF1 original PC vs PC projected onto ucomp EOF1')
                plot_dir_EOF_proj = f'{plot_dir_EOF}/proj_comparisons/{hemisphere}_hemisphere/{time_frame}/'
                if not os.path.isdir(plot_dir_EOF_proj):
                    os.makedirs(plot_dir_EOF_proj)
                plt.savefig(f'{plot_dir_EOF_proj}/{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}_vs_orig_PC{prop_nan_str}.pdf')
