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
    main_path = Path('/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical')
    base_save_path = Path('/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/hist_processed')
    
    if not main_path.exists():
        logger.error(f"Main data path does not exist: {main_path}")
        raise FileNotFoundError(f"Main data path not found: {main_path}")
    
    base_save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base output directory ensured: {base_save_path}")
    
    return main_path, base_save_path

def get_model_list(main_path):
    """Get list of models, excluding EFP_data directory"""
    try:
        main_files = [f for f in os.listdir(main_path)]
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
        
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        
        ds = xr.open_mfdataset(model_path, combine='by_coords', chunks={'time':30}, decode_times=time_coder)
        
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
        ds.close()  # Close the original dataset to free up resources
        
        logger.info(f"Successfully resampled {model_name} to monthly means")
        return ds_mon
        
    except Exception as e:
        logger.error(f"Error during resampling for model {model_name}: {e}")
        raise


def process_model(model, main_path, base_save_path, year_range=None):
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
            
            try:
                # Calculate experiment length
                if experiment in ['1850_2014', '1850_2015']:
                    experiment_length = '1850_2014'
                elif experiment in ['1950_2014', '1950_2015']:
                    experiment_length = '1950_2014'
                else:
                    logger.warning(f"Unexpected experiment name format: {experiment}, using raw name as length")
                    experiment_length = experiment
                
                if experiment_length is None:
                    logger.error(f"Could not determine experiment length for {experiment}, skipping")
                    continue
                
                if year_range is not None:
                    logger.info(f"Using specified year range: {year_range[0]} - {year_range[1]} for experiment {experiment}")
                    experiment_length = f'{year_range[0]}_{year_range[1]}'
                
                
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
                        
                        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                        with xr.open_dataset(daily_data_dir / daily_file, chunks={'time': 30}, 
                                            decode_times=time_coder) as ds:
                            
                            # SUBSET TO TARGET TIME WINDOW
                            if model in ['HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'UKESM1-0-LL']:
                                logger.info(f"Applying special time window for {model}: 1950-01-01 to 2014-12-30")
                                ds_subset = ds.sel(time=slice('1950-01-01', '2014-12-30'))
                            else:
                                ds_subset = ds.sel(time=slice('1950-01-01', '2014-12-31'))
                            
                            # Skip if no data in window
                            if len(ds_subset.time) == 0:
                                logger.warning(f"Skipping {daily_file}: no data in 1950-2014 window")
                                continue
                            
                            # Extract dates using string conversion (handles all datetime types)
                            time_start_str = str(ds_subset.time[0].values)
                            time_end_str = str(ds_subset.time[-1].values)

                            # Parse YYYYMMDD from ISO format string (works for cftime, datetime64, etc.)
                            start_str = time_start_str[:10].replace('-', '')  # "YYYY-MM-DD" -> "YYYYMMDD"
                            end_str = time_end_str[:10].replace('-', '')
                            year_str = f'{start_str}_{end_str}'
                            
                            # Log the processing
                            n_times = len(ds_subset.time)
                            logger.info(f"Input file:  {daily_file}")
                            logger.info(f"Processing {n_times} timesteps: {time_start_str} to {time_end_str}")
                            logger.info(f"Output file: {year_str}_dm_uvt_epfluxes.nc")
                            
                            # Create output filename from ACTUAL data
                            save_file_name = f'{year_str}_dm_uvt_epfluxes.nc'
                            
                            # If file already exists, skip processing
                            output_file = save_path / save_file_name
                            if output_file.exists():
                                logger.info(f"Output file already exists, skipping: {output_file}")
                                files_processed += 1
                                continue
                            
                            # Use subsetted dataset for all processing
                            ds = ds_subset
                            
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
    
    YEAR_RANGE = (1950, 2014)  # Default year range to use for processing and output naming
    
    try:
        # Setup paths
        main_path, base_save_path = setup_paths()
        
        # Get model list
        models = get_model_list(main_path)
        
        # Process each model
        success_count = 0
        for i, model in enumerate(models, 1):
            logger.info(f"Processing model {i}/{len(models)}: {model}")
            
            if process_model(model, main_path, base_save_path, year_range=YEAR_RANGE):
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