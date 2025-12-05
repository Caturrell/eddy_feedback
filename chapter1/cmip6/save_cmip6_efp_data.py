import xarray as xr
import numpy as np
import os
import logging
import sys
from pathlib import Path

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
    """Process EFP data for a single model, handling multiple experiments"""
    model_path = main_path / model
    
    try:
        # Get all experiments for this model
        experiments = sorted(os.listdir(model_path))
        logger.info(f"Found {len(experiments)} experiment(s) for model {model}: {experiments}")
        
        success_count = 0
        
        # Loop over all experiments
        for experiment in experiments:
            try:
                # Calculate experiment length
                experiment_length = calculate_experiment_length(experiment)
                
                if experiment_length is None:
                    logger.error(f"Could not determine experiment length for {experiment}, skipping")
                    continue
                
                # Create the appropriate subdirectory
                save_path = base_save_path / experiment_length
                save_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using output directory: {save_path}")
                
                efp_file = model_path / experiment / '6hrPlevPt' / 'efp_500hPa.nc'
                
                if not efp_file.exists():
                    logger.warning(f"File not found: {efp_file}")
                    continue
                
                # Process the data
                logger.info(f"Processing model: {model}, experiment: {experiment} (length: {experiment_length})")
                
                with xr.open_dataset(efp_file) as ds:
                    # Extract EFP variables
                    efp_vars = [var for var in ds.data_vars if var.startswith('efp_ucomp_div1_QG_')]
                    
                    if not efp_vars:
                        logger.warning(f"No EFP variables found in {efp_file}")
                        continue
                    
                    ds_efp = ds[efp_vars]
                    
                    # Save the processed data with both model and experiment in filename
                    output_file = save_path / f'{model}_{experiment}_efp_500hPa.nc'
                    ds_efp.to_netcdf(output_file)
                    logger.info(f"Successfully saved EFP data for {model} - {experiment} to {output_file}")
                    
                    # Log some basic info about the saved data
                    logger.debug(f"Variables saved: {list(ds_efp.data_vars.keys())}")
                    logger.debug(f"Data dimensions: {dict(ds_efp.dims)}")
                    
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing experiment {experiment} for model {model}: {e}", exc_info=True)
                continue
        
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