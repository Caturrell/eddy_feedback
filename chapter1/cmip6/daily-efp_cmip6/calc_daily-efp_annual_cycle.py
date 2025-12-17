"""
Docstring for chapter1.cmip6.calc_efp_annual_cycle

Script copied from: chapter1/annual_cycle/pamip_annual_cycle.py

Improved version with:
- Dataset validation
- Better error handling
- Memory-efficient processing
- Robust scalar extraction
- Dimension validation
"""


import xarray as xr
import os
import numpy as np
import warnings
import logging
import json
from pathlib import Path
import pandas as pd
import glob

import functions.eddy_feedback as ef
import functions.data_wrangling as dw
import functions.aos_functions as aos

# Set up logger at module level
logger = logging.getLogger(__name__)


def validate_dataset(ds, model_name):
    """Validate that dataset contains all required variables and dimensions."""
    required_vars = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3', 'ubar']
    missing_vars = [var for var in required_vars if var not in ds]
    
    if missing_vars:
        logger.error(f"Model {model_name} missing variables: {missing_vars}")
        return False
    
    if 'time' not in ds.dims:
        logger.error(f"Model {model_name} missing required dimension 'time'. Found: {list(ds.dims)}")
        return False
    
    logger.info(f"Dataset validation passed for model: {model_name}")
    return True


def resample_to_monthly(base_path, model_name):
    """Resample daily data to monthly means and calculate zonal mean wind."""
    logger.info(f"Resampling data to monthly for model: {model_name}")
    
    try:
        
        logger.info(f"Loading daily data for model: {model_name}")
        
        model_path = os.path.join(base_path, model_name)
        model_files = glob.glob(os.path.join(model_path, '*.nc'))
        
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_mfdataset(model_files, combine='by_coords', decode_times=time_coder, chunks={'time': 31})
        
        # run data through checker
        ds = dw.data_checker1000(ds, check_vars=False)
        
        # Calculate ubar (zonal mean wind)
        if 'ucomp' not in ds:
            raise ValueError(f"Variable 'ucomp' not found in dataset for model {model_name}")
        
        logger.info("Calculating zonal mean wind (ubar)...")
        ds['ubar'] = ds['ucomp'].mean('lon')
        
        # Resample to monthly means
        logger.info("ubar calculated. Resampling to monthly means...")
        ds_mon = ds.resample(time='1ME').mean()
        
        logger.info(f"Successfully resampled {model_name} to monthly means")
        return ds_mon
        
    except Exception as e:
        logger.error(f"Error during resampling for model {model_name}: {e}")
        raise


def seasonal_mean(ds, months, take_mean=False):
    """Calculate seasonal means from monthly data."""
    logger.info(f"Computing seasonal mean for months: {months}, take_mean={take_mean}")
    
    # Pre-processing checks
    if not (isinstance(months, list) and all(isinstance(m, int) and 1 <= m <= 12 for m in months)):
        raise ValueError(f"`months` must be a list of 3 integers between 1–12. Got: {months}")
    if len(months) != 3:
        raise ValueError(f"`months` must have exactly 3 elements. Got: {months}")

    ds_season = ds.sel(time=ds['time'].dt.month.isin(months))

    def assign_season_year(time):
        """Assign year to season, accounting for DJF crossing year boundary."""
        year = time.dt.year
        month = time.dt.month
        # For DJF, December belongs to the following year's season
        if months[0] == 12:
            year = xr.where(month == 12, year + 1, year)
        return year

    season_year = assign_season_year(ds_season['time'])
    ds_season.coords['season_year'] = ('time', season_year.data)

    result = ds_season.groupby('season_year').mean('time')
    result = result.rename({'season_year': 'time'})

    if take_mean:
        logger.info("Returning mean over all seasons.")
        return result.mean('time')
    else:
        return result


def calculate_efp_annual_cycle(ds, months, calc_south_hemis=False, which_div1=None):
    """Calculate Eddy Feedback Parameter for a specific season and hemisphere."""
    hemi = 'Southern' if calc_south_hemis else 'Northern'
    logger.info(f"Calculating EFP annual cycle for {hemi} Hemisphere, div1: {which_div1}, months: {months}")
    
    # Validate that required variable exists
    if which_div1 not in ds:
        raise ValueError(f"Variable '{which_div1}' not found in dataset")
    
    # Select hemisphere
    if calc_south_hemis:
        ds = ds.sel(lat=slice(-90, 0))
        efp_lat_slice = slice(-75, -25)
    else:
        ds = ds.sel(lat=slice(0, 90))
        efp_lat_slice = slice(25, 75)
    
    # Calculate seasonal means
    ds = seasonal_mean(ds, months=months)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = xr.corr(ds[which_div1], ds.ubar, dim='time').load()**2

        # Select latitude range
        corr = corr.sel(lat=efp_lat_slice)
        
        # Select pressure level
        corr = corr.sel(level=500., method='nearest')
        actual_level = float(corr.level.values)
        
        if abs(actual_level - 500) > 50:
            logger.warning(f"⚠️ Nearest level {actual_level} is far from 500 hPa!")

        # Calculate area-weighted mean
        weights = np.cos(np.deg2rad(corr.lat))
        efp = corr.weighted(weights).mean('lat')

        efp_value = round(float(efp.values.item()), 4)
        
        # **CHECK FOR SUSPICIOUS VALUES**
        if efp_value == 1.0:
            logger.warning(f"⚠️ EFP = 1.0 (perfect correlation - likely data issue!)")
        elif np.isnan(efp_value):
            logger.warning(f"⚠️ EFP = NaN")
            
        logger.info(f"EFP = {efp_value}")
        return efp_value
    
    except Exception as e:
        logger.error(f"Error during EFP calculation: {e}")
        raise RuntimeError(f"Error during Eddy Feedback Parameter calculation: {e}")


def compute_and_save_efp_seasonal(dataset, output_dir, time_period, model):
    """Compute EFP for all seasons and configurations, save to JSON."""
    logger.info(f"Starting computation of seasonal EFPs for model: {model}")
    logger.info(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Define all 3-month seasons
    season_month_dict = {
        'DJF': [12, 1, 2], 'JFM': [1, 2, 3], 'FMA': [2, 3, 4], 'MAM': [3, 4, 5],
        'AMJ': [4, 5, 6], 'MJJ': [5, 6, 7], 'JJA': [6, 7, 8], 'JAS': [7, 8, 9],
        'ASO': [8, 9, 10], 'SON': [9, 10, 11], 'OND': [10, 11, 12], 'NDJ': [11, 12, 1]
    }

    # Define all configurations (hemisphere and wavenumber combinations)
    configs = [
        ('efp_nh',      'div1_QG',     False),
        ('efp_nh_123',  'div1_QG_123', False),
        ('efp_nh_gt3',  'div1_QG_gt3', False),
        ('efp_sh',      'div1_QG',     True),
        ('efp_sh_123',  'div1_QG_123', True),
        ('efp_sh_gt3',  'div1_QG_gt3', True),
    ]

    # Validate dataset before processing
    if not validate_dataset(dataset, model):
        logger.error(f"Dataset validation failed for model {model}. Skipping.")
        return None

    # Dictionary to hold all results
    all_results = {}

    for key, which_div1, calc_south_hemis in configs:
        logger.info(f"Processing configuration: {key}")
        result = {}

        for season, months in season_month_dict.items():
            logger.info(f"-> Computing result for config={key}, season={season}, months={months}")
            
            try:
                # Calculate EFP for the current season
                efp = calculate_efp_annual_cycle(
                    dataset,
                    months=months,
                    calc_south_hemis=calc_south_hemis,
                    which_div1=which_div1
                )

                result[season] = {
                    "efp": efp,
                    "months": months
                }
            
            except Exception as e:
                logger.error(f"Failed to calculate EFP for {key}, {season}: {e}")
                result[season] = {
                    "efp": None,
                    "months": months,
                    "error": str(e)
                }

        all_results[key] = result

    # Save all results into a single JSON file
    json_path = os.path.join(output_dir, f"{model}_efp_results_CMIP6_piControl_{time_period}.json")
    
    try:
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"✅ Saved all results to JSON: {json_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save JSON file: {e}")
        raise

    logger.info("Completed all configurations.")
    return all_results


if __name__ == "__main__":
    
    # Get the directory where this script is located
    import sys
    script_dir = Path(__file__).parent

    # Configure logging - save log file in the same directory as the script
    log_file = script_dir / 'calc_efp_annual_cycle.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger.info(f"Log file location: {log_file}")
    logger.info("="*70)
    logger.info("Starting EFP Annual Cycle Processing Script")
    logger.info("="*70)
    
    #==============================================================================================
    
    ## SELECT TIME PERIOD TO PROCESS ##
    time_period = '100y'  # Options: 30y or '100y'
    
    cmip_path = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit'
    cmip_exp_path = os.path.join(cmip_path, time_period)
    
    logger.info(f"Time period: {time_period}")
    logger.info(f"CMIP experiment path: {cmip_exp_path}")
    
    #==============================================================================================
    
    ### Calculate monthly means for all models (if not already done) ###
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Monthly Resampling")
    logger.info("="*70)
    
    # Create save path for monthly data
    save_mon_path = Path(cmip_exp_path) / 'mon_avg_daily'
    save_mon_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of models
    daily_avg_path = Path(cmip_exp_path) / 'daily_averages'
    
    if not daily_avg_path.exists():
        logger.error(f"Daily averages path does not exist: {daily_avg_path}")
        sys.exit(1)
    
    model_list = [d for d in daily_avg_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(model_list)} models to process")
    
    for idx, model_dir in enumerate(model_list, 1):
        
        model_name = model_dir.name
        
        
        logger.info(f"\n[{idx}/{len(model_list)}] Processing model: {model_name}")
        
        # Check if monthly data already exists
        save_file_name = f'{model_name}_{time_period}_mm_uvt_ubar_epf.nc'
        out_file = save_mon_path / save_file_name
        
        if out_file.exists():
            logger.info(f"✓ Monthly data already exists, skipping resampling: {out_file.name}")
            continue
        
        try:
            # Resample to monthly mean
            resampled_ds = resample_to_monthly(
                base_path=str(daily_avg_path),
                model_name=model_name
            )
            
            # Save resampled dataset
            resampled_ds.to_netcdf(out_file)
            logger.info(f"✅ Saved monthly data: {out_file}")
            
            # Clean up
            resampled_ds.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to process {model_name}: {e}")
            logger.error(f"Continuing with next model...")
            continue
    
    logger.info("\n" + "="*70)
    logger.info("Monthly resampling completed for all models")
    logger.info("="*70)
    
    #==============================================================================================
    
    ### Calculate seasonal EFPs for all models ###
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Seasonal EFP Calculations")
    logger.info("="*70)
    
    # Get list of models with monthly data
    model_files = sorted(save_mon_path.glob('*.nc'))
    logger.info(f"Found {len(model_files)} monthly datasets to process")
    
    if len(model_files) == 0:
        logger.warning("No monthly datasets found. Exiting.")
        sys.exit(0)
    
    output_dir = Path(f'/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/data/{time_period}/daily')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process models one at a time to manage memory
    successful_models = 0
    failed_models = []
    
    for idx, model_file in enumerate(model_files, 1):
        # Extract model name from filename (more robust extraction)
        model_name = model_file.stem.split('_')[0]
        logger.info(f"\n[{idx}/{len(model_files)}] Processing model: {model_name}")
        logger.info(f"Loading dataset: {model_file.name}")
        
        try:
            # Load dataset
            ds = xr.open_mfdataset(model_file, chunks={'time': 31}, combine='nested')
            logger.info(f"Dataset loaded successfully. Dimensions: {dict(ds.sizes)}")
            
            # Create model-specific output directory
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute and save EFP results
            results = compute_and_save_efp_seasonal(
                dataset=ds,
                output_dir=str(model_output_dir),
                time_period=time_period,
                model=model_name
            )
            
            if results is not None:
                successful_models += 1
                logger.info(f"✅ Successfully processed {model_name}")
            else:
                failed_models.append(model_name)
                logger.warning(f"⚠️ Processing completed with issues for {model_name}")
            
            # Clean up - close dataset to free memory
            ds.close()
            del ds
            
        except Exception as e:
            logger.error(f"❌ Failed to process {model_name}: {e}")
            logger.error(f"Continuing with next model...")
            failed_models.append(model_name)
            continue
    
    #==============================================================================================
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total models: {len(model_files)}")
    logger.info(f"Successful: {successful_models}")
    logger.info(f"Failed: {len(failed_models)}")
    
    if failed_models:
        logger.warning(f"Failed models: {', '.join(failed_models)}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*70)
    logger.info("Script completed successfully!")