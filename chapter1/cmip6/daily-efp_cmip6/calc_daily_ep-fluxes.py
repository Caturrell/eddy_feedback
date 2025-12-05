import xarray as xr
import numpy as np
import os
import logging
import sys
from pathlib import Path
import glob

import functions.eddy_feedback as ef
import functions.aos_functions as aos
import functions.data_wrangling as dw

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Configure logging - save log file in the same directory as the script
log_file = script_dir / 'efp_processing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Log file location: {log_file}")

def setup_paths():
    """Setup and validate input/output paths"""
    main_path = Path('/gws/nopw/j04/arctic_connect/sthomson/efp_6hourly_processed_data/1000hPa_100hPa_slice_inner2')
    base_save_path = Path('/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit')
    
    if not main_path.exists():
        logger.error(f"Main data path does not exist: {main_path}")
        raise FileNotFoundError(f"Main data path not found: {main_path}")
    
    base_save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base output directory ensured: {base_save_path}")
    
    return main_path, base_save_path

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
    
def resample_to_monthly(base_path, model_name):
    """Resample daily data to monthly means and calculate zonal mean wind."""
    logger.info(f"Resampling data to monthly for model: {model_name}")
    
    try:
        
        logger.info(f"Loading daily data for model: {model_name}")
        model_path = glob.glob(os.path.join(base_path, model_name, '*_dm_uvt_epfluxes.nc'))
        
        ds = xr.open_mfdataset(model_path, combine='by_coords', chunks={'time':12})
        
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

def calculate_experiment_length(experiment_name):
    """
    Calculate experiment length from experiment name.
    Assumes format like 'YYYY_YYYY' where the difference gives the length.
    Returns '30y' or '100y'.
    """
    try:
        # Split by underscore and extract the two years
        parts = experiment_name.split('_')
        
        # Find the two numeric parts (years)
        years = [int(part) for part in parts if part.isdigit()]
        
        if len(years) < 2:
            logger.warning(f"Could not extract two years from experiment name: {experiment_name}")
            return None
        
        # Calculate the difference between the two years
        year_diff = abs(years[1] - years[0])
        
        # Determine if it's 30 or 100 years
        if year_diff in [29, 30]:
            return '30y'
        elif year_diff in [99, 100]:
            return '100y'
        else:
            logger.warning(f"Unexpected year difference {year_diff} for experiment: {experiment_name}")
            return 'other'
            
    except Exception as e:
        logger.error(f"Error calculating experiment length for {experiment_name}: {e}")
        return None

def process_model(model, main_path, base_save_path):
    """Process daily data for a single model, handling multiple experiments"""
    model_path = main_path / model
    
    try:
        # Get all experiments for this model
        experiments = sorted(os.listdir(model_path))
        logger.info(f"Found {len(experiments)} experiment(s) for model {model}: {experiments}")
        
        success_count = 0
        processed_lengths = set()  # Track which experiment lengths had successful processing
        
        # Loop over all experiments
        for experiment in experiments:
            
            
            # SKIP MODELS THAT WERE BEING KILLED
            # if model in ['GFDL-CM4', 'IPSL-CM6A-LR', 'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR',
            #              'MPI-ESM1-2-LR', 'NorESM2-LM']:
            #     logger.info(f"SKIPPING {model} DUE TO UNKNOWN DATA ISSUE")
            #     continue
            
            
            try:
                # Calculate experiment length
                experiment_length = calculate_experiment_length(experiment)
                
                if experiment_length == '100y':
                    logger.info(f"Skipping 100-year experiment: {experiment}")
                    continue
                
                if experiment_length is None:
                    logger.error(f"Could not determine experiment length for {experiment}, skipping")
                    continue
                
                # Create the appropriate subdirectory
                save_path = base_save_path / experiment_length / 'daily_averages' / model
                save_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using output directory: {save_path}")
                
                # Locate daily data files
                daily_data_dir = model_path / experiment / '6hrPlevPt' / 'yearly_data'
                
                if not daily_data_dir.exists():
                    logger.error(f"Data directory not found: {daily_data_dir}")
                    continue
                    
                daily_data_files = [f for f in os.listdir(daily_data_dir) if f.endswith('_daily_averages.nc')]
                
                if not daily_data_files:
                    logger.warning(f"No daily data files found for {model}/{experiment}")
                    continue
                
                # Track file processing
                files_processed = 0
                files_failed = []
                
                logger.info(f"Processing model: {model}, experiment: {experiment} (length: {experiment_length})")
                
                for daily_file in daily_data_files:
                    
                    if not (daily_data_dir / daily_file).exists():
                        logger.warning(f"File not found: {daily_file}")
                        files_failed.append(daily_file)
                        continue
                    
                    try:
                        logger.info(f"Processing daily data file: {daily_file}")
                        
                        # extract years and set save file name
                        years = daily_file.split('_')[0:2]
                        year_str = '_'.join(years)
                        save_file_name = f'{year_str}_dm_uvt_epfluxes.nc'
                        
                        # If file already exists, skip processing
                        output_file = save_path / save_file_name
                        if output_file.exists():
                            logger.info(f"Output file already exists, skipping: {output_file}")
                            files_processed += 1
                            continue
                        
                        
                        with xr.open_dataset(daily_data_dir / daily_file, chunks={'time': 12}) as ds:
                            
                            # ds = ds.rename({'pfull': 'level'})
                            
                            # subset dataset to required variables
                            vars_uvt = ['ucomp', 'vcomp', 'temp']
                            ds = ds[vars_uvt]
                            
                            ucomp = ds['ucomp']
                            vcomp = ds['vcomp']
                            temp = ds['temp']
                            
                            # first calculate full EP fluxes
                            logger.info("Computing full EP fluxes")
                            ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp)
                            
                            # save variables to dataset
                            ds['ep1_QG'] = (ep1.dims, ep1.values)
                            ds['ep2_QG'] = (ep2.dims, ep2.values)
                            ds['div1_QG'] = (div1.dims, div1.values)
                            ds['div2_QG'] = (div2.dims, div2.values)
                            
                            # Calculate for k123
                            logger.info("Computing EP fluxes for waves 1,2,3")
                            ep1_k123, ep2_k123, div1_k123, div2_k123 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp, wave=[1, 2, 3])
                            
                            # save k123 variables to dataset
                            ds['ep1_QG_123'] = (ep1_k123.dims, ep1_k123.values)
                            ds['ep2_QG_123'] = (ep2_k123.dims, ep2_k123.values)
                            ds['div1_QG_123'] = (div1_k123.dims, div1_k123.values)
                            ds['div2_QG_123'] = (div2_k123.dims, div2_k123.values)  
                            
                            # Decompose >k=3
                            logger.info(f"Decomposing EP fluxes and saving data...")
                            for var_name_to_decompose in ['ep1_QG', 'ep2_QG', 'div1_QG', 'div2_QG']:
                                ds[f'{var_name_to_decompose}_gt3'] = ds[f'{var_name_to_decompose}'] - ds[f'{var_name_to_decompose}_123']
                                
                            
                            ds.to_netcdf(output_file)
                            logger.info(f"Successfully saved daily EP fluxes for {year_str} to {output_file}")
                            
                            # Log some basic info about the saved data
                            logger.debug(f"Variables saved: {list(ds.data_vars.keys())}")
                            logger.debug(f"Data dimensions: {dict(ds.dims)}")
                        
                        files_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing file {daily_file}: {e}", exc_info=True)
                        files_failed.append(daily_file)
                        continue
                
                # Log summary for this experiment
                logger.info(f"Experiment {experiment}: {files_processed}/{len(daily_data_files)} files processed successfully")
                if files_failed:
                    logger.warning(f"Failed files for {experiment}: {files_failed}")
                
                # Count as success if at least one file was processed
                if files_processed > 0:
                    success_count += 1
                    processed_lengths.add(experiment_length)  # Track this experiment length
                
            except Exception as e:
                logger.error(f"Error processing experiment {experiment} for model {model}: {e}", exc_info=True)
                continue
        
        # After processing all experiments, perform monthly resampling for each experiment length
        if processed_lengths:
            logger.info("\n" + "="*70)
            logger.info(f"MONTHLY RESAMPLING for {model}")
            logger.info("="*70)
            
            for experiment_length in sorted(processed_lengths):
                try:
                    # Setup monthly save path
                    save_mon_path = base_save_path / experiment_length / 'mon_avg_daily'
                    save_mon_path.mkdir(parents=True, exist_ok=True)
                    
                    # Define monthly output file name
                    save_file_name = f'{model}_{experiment_length}_mm_uvt_ubar_epf.nc'
                    out_file = save_mon_path / save_file_name
                    
                    # Check if monthly data already exists
                    if out_file.exists():
                        logger.info(f"✓ Monthly data already exists for {experiment_length}, skipping: {out_file.name}")
                    else:
                        # Resample to monthly mean - this will load ALL daily files from daily_averages/{model}/
                        logger.info(f"Starting monthly resampling for {model} ({experiment_length})...")
                        daily_avg_path = base_save_path / experiment_length / 'daily_averages'
                        
                        resampled_ds = resample_to_monthly(
                            base_path=str(daily_avg_path),
                            model_name=model
                        )
                        
                        # Save resampled dataset
                        resampled_ds.to_netcdf(out_file)
                        logger.info(f"✅ Saved monthly data: {out_file}")
                        
                        # Clean up
                        resampled_ds.close()
                        
                except Exception as e:
                    logger.error(f"❌ Failed to create monthly data for {model} ({experiment_length}): {e}", exc_info=True)
                    logger.info("Continuing with next length...")
            
            logger.info("="*70 + "\n")
        
        # Return True if at least one experiment was processed successfully
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error processing model {model}: {e}", exc_info=True)
        return False

def main():
    """Main processing function"""
    logger.info("Starting EFP data processing pipeline")
    
    try:
        # Setup paths
        main_path, base_save_path = setup_paths()
        
        # Get model list
        models = get_model_list(main_path)
        
        # Process each model
        success_count = 0
        for i, model in enumerate(models, 1):
            logger.info(f"Processing model {i}/{len(models)}: {model}")
            
            if process_model(model, main_path, base_save_path):
                success_count += 1
        
        # Summary
        logger.info(f"Processing completed. Successfully processed {success_count}/{len(models)} models")
        
        if success_count < len(models):
            logger.warning(f"Failed to process {len(models) - success_count} models")
            
    except Exception as e:
        logger.error(f"Fatal error in main processing loop: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()