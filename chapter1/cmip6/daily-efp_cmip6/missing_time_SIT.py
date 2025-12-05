"""
Simple plot of daily averaged climate data for CMIP6 models.
"""

import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
import warnings
import logging

warnings.filterwarnings('ignore', category=xr.SerializationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Import custom functions
import functions.data_wrangling as dw


def get_model_list(main_path):
    """Get list of models, excluding EFP_data directory"""
    try:
        main_files = [f for f in os.listdir(main_path) if f not in ['EFP_data', 'EFP_data_mk2']]
        models = sorted(main_files)  # Sort for consistent processing order
        logger.info(f"Found {len(models)} models to process")
        return models
    except Exception as e:
        logger.error(f"Error reading model list from {main_path}: {e}")
        raise



# MAIN EXECUTION FOR SIT DATA
# def main():
#     """Main execution function - multiple models."""
#     base_path = '/gws/nopw/j04/arctic_connect/sthomson/efp_6hourly_processed_data/1000hPa_100hPa_slice_inner2'
    
#     logger.info("Starting CMIP6 daily data plotting")
    
#     models = get_model_list(base_path)
    
#     fig, axes = plt.subplots(len(models), 1, figsize=(12, 4*len(models)))
#     if len(models) == 1:
#         axes = [axes]
    
#     for i, model in enumerate(models):
#         logger.info(f"Processing {model} ({i+1}/{len(models)})")
        
#         try:
#             experiments = os.listdir(os.path.join(base_path, model))
#             experiment = experiments[0]
            
#             model_path = os.path.join(base_path, model, f'{experiment}/6hrPlevPt/yearly_data')
#             files = glob.glob(os.path.join(model_path, '*_daily_averages.nc'))
            
            
#             logger.info(f"{model}: Found {len(files)} files")
            
#             ds = xr.open_mfdataset(files, combine='by_coords', chunks={'time': 12}, compat='override')
#             ubar = ds['ucomp'].sel(pfull=500., method='nearest').sel(lat=slice(25, 75)).mean('lon')
#             ubar.plot(ax=axes[i])
#             axes[i].set_title(f'{model} - ubar (500 hPa)')
#             ds.close()
            
#             logger.info(f"{model}: Completed successfully")
            
#         except Exception as e:
#             logger.error(f"Error processing {model}: {e}")
    
#     plt.tight_layout()
#     plt.savefig('cmip6_daily_ubar_500hPa_with_override.png')
    
#     logger.info("Plotting complete")


# MAIN EXECUTION FOR MY DAILY DATA
def main():
    """Main execution function - multiple models."""
    base_path = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit/30y/daily_averages'
    
    logger.info("Starting CMIP6 daily data plotting for my daily data")
    
    models = get_model_list(base_path)
    
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4*len(models)))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        logger.info(f"Processing {model} ({i+1}/{len(models)})")
        
        try:
            
            model_path = os.path.join(base_path, model)
            files = glob.glob(os.path.join(model_path, '*_dm_uvt_epfluxes.nc'))
            
            
            logger.info(f"{model}: Found {len(files)} files")
            
            ds = xr.open_mfdataset(files, combine='by_coords', chunks={'time': 12})#, compat='override')
            ubar = ds['ucomp'].sel(pfull=500., method='nearest').sel(lat=slice(25, 75)).mean('lon')
            ubar.plot(ax=axes[i])
            axes[i].set_title(f'{model} - ubar (500 hPa)')
            ds.close()
            
            logger.info(f"{model}: Completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing {model}: {e}")
    
    plt.tight_layout()
    plt.savefig('my_data_cmip6_daily_ubar_500hPa.png')
    
    logger.info("Plotting complete")


if __name__ == '__main__':
    main()
