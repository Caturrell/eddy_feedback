"""
Plot daily and monthly averaged climate data for CMIP6 models.
Compares models with different EFP classifications.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import logging
import glob

warnings.filterwarnings('ignore', category=xr.SerializationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import custom functions
import functions.eddy_feedback as ef
import functions.data_wrangling as dw


def load_monthly_data(data_path):
    """Load monthly averaged data (one file per model)."""
    file_list = [file for file in os.listdir(data_path) if file.endswith('.nc')]
    datasets = {}
    
    for file in file_list:
        logger.info(f"Loading monthly: {file}")
        file_path = os.path.join(data_path, file)
        model_name = file.split('_')[0]
        ds = xr.open_dataset(file_path, chunks={'time': 12})
        datasets[model_name] = ds
    
    return datasets


def process_seasonal_data(ds, variables=['ubar', 'div1_QG'],
                          level=500.,
                          lat_range=(25, 75)):
    """
    Process climate data for seasonal analysis.
    """
    # Subset data
    ds_out = ds[variables]
    ds_out = ds_out.sel(level=level, method='nearest').sel(lat=slice(lat_range[0], lat_range[1]))
    logger.debug(f'Data shape before time slicing: {ds_out.sizes}')
    
    return ds_out


def plot_all_models(datasets, model_list, title_prefix, variables=['ubar', 'div1_QG']):
    """Plot all models from a classification on one figure - each model gets its own row."""
    
    n_models = len(model_list)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    
    # Handle case where there's only one model (axes won't be 2D)
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for i, model in enumerate(model_list):
        logger.info(f"Processing model: {model}")
        if model not in datasets:
            logger.warning(f"{model} not found in datasets")
            continue
            
        ds_processed = process_seasonal_data(datasets[model], variables=variables)
        
        # Plot ubar in first column
        ds_processed[variables[0]].plot(ax=axes[i, 0])
        axes[i, 0].set_title(f'{model} - {variables[0]}')
        
        # Plot div1_QG in second column
        ds_processed[variables[1]].plot(ax=axes[i, 1])
        axes[i, 1].set_title(f'{model} - {variables[1]}')
    
    fig.suptitle(title_prefix, fontsize=14, y=1.00)
    plt.tight_layout()
    return fig


# def plot_all_models_batch(data_path, model_list, title_prefix, 
#                           variables=['ubar', 'div1_QG'], 
#                           save_path=None):
#     """
#     Plot models by loading them one at a time to avoid memory issues.
#     """
#     n_models = len(model_list)
#     fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    
#     if n_models == 1:
#         axes = axes.reshape(1, -1)
    
#     for i, model in enumerate(model_list):
#         logger.info(f"Processing model {i+1}/{n_models}: {model}")
        
#         # Load just this one model
#         model_path = os.path.join(data_path, model)
#         file_paths = glob.glob(os.path.join(model_path, '*_dm_uvt_epfluxes.nc'))
        
#         try:
#             ds = xr.open_mfdataset(
#                 file_paths, 
#                 combine='by_coords',
#                 chunks={'time': 12}
#             )
            
#             ds = dw.data_checker1000(ds, check_vars=False)
            
#             if 'ubar' not in ds:
#                 logger.info(f"Calculating ubar for {model}")
#                 ds['ubar'] = ds.ucomp.mean(dim='lon')
            
#             # Process and plot
#             ds_processed = process_seasonal_data(ds, variables=variables)
            
#             # Plot first variable
#             ds_processed[variables[0]].plot(ax=axes[i, 0])
#             axes[i, 0].set_title(f'{model} - {variables[0]}')
            
#             # Plot second variable
#             ds_processed[variables[1]].plot(ax=axes[i, 1])
#             axes[i, 1].set_title(f'{model} - {variables[1]}')
            
#             # Explicitly close and clear memory
#             ds.close()
#             del ds, ds_processed
            
#         except Exception as e:
#             logger.error(f"Failed to process {model}: {e}")
#             continue
    
#     fig.suptitle(title_prefix, fontsize=14, y=1.00)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved plot to {save_path}")
    
#     return fig


def main():
    """Main execution function."""
    
    # Define paths
    mon_avg_path = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit/100y/mon_avg_daily'
    daily_path = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit/100y/daily_averages'
    save_plot_path = '/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/plots/missing_time_data'
    
    # Model list
    models = [
        'AWI-ESM-1-1-LR',
        'CMCC-CM2-SR5',
        'CMCC-ESM2',
        'CanESM5-1',
        'EC-Earth3',
        'EC-Earth3-CC',
        'EC-Earth3-LR',
        'EC-Earth3-Veg',
        'EC-Earth3-Veg-LR',
        'IPSL-CM5A2-INCA',
        'IPSL-CM6A-LR',
        'IPSL-CM6A-MR1',
        'MPI-ESM-1-2-HAM',
        'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR',
        'MRI-ESM2-0'
    ]
    
    # Variables to plot
    vars_to_plot = ['ubar', 'div1_QG']
    
    # Load monthly averaged data (smaller, okay to load all at once)
    logger.info("=== Loading Monthly Averaged Data ===")
    datasets_monthly = load_monthly_data(mon_avg_path)
    
    # Plot monthly data
    logger.info("=== Plotting Monthly Data ===")
    fig1 = plot_all_models(
        datasets_monthly, 
        models, 
        'CMIP6 Models (Monthly Averaged)', 
        variables=vars_to_plot
    )
    plt.savefig(
        f'{save_plot_path}/100y_models_monthly_{vars_to_plot[0]}_{vars_to_plot[1]}.png', 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close(fig1)
    
    # Clean up monthly data
    for ds in datasets_monthly.values():
        ds.close()
    del datasets_monthly
    
    # Plot daily data - process one model at a time
    # logger.info("=== Plotting Daily Data ===")
    # fig2 = plot_all_models_batch(
    #     daily_path, 
    #     models, 
    #     'CMIP6 Models (Daily Averages)',
    #     save_path=f'{save_plot_path}/100y_models_daily_{vars_to_plot[0]}_{vars_to_plot[1]}.png',
    #     variables=vars_to_plot
    # )
    # plt.close(fig2)
    
    logger.info("=== Complete ===")


if __name__ == '__main__':
    main()