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
    save_path = Path('/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit')
    
    if not main_path.exists():
        logger.error(f"Main data path does not exist: {main_path}")
        raise FileNotFoundError(f"Main data path not found: {main_path}")
    
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {save_path}")
    
    return main_path, save_path

def get_model_list(main_path):
    """Get list of models, excluding EFP_data directory"""
    try:
        main_files = [f for f in os.listdir(main_path) if f != 'EFP_data']
        models = sorted(main_files)  # Sort for consistent processing order
        logger.info(f"Found {len(models)} models to process")
        return models
    except Exception as e:
        logger.error(f"Error reading model list from {main_path}: {e}")
        raise

def process_model(model, main_path, save_path):
    """Process EFP data for a single model"""
    model_path = main_path / model
    
    try:
        # Check experiment directory
        experiments = os.listdir(model_path)
        if len(experiments) > 1:
            error_msg = f"More than one experiment found for model {model}: {experiments}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        experiment = experiments[0]
        efp_file = model_path / experiment / '6hrPlevPt' / 'efp_500hPa.nc'
        
        if not efp_file.exists():
            logger.warning(f"File not found: {efp_file}")
            return False
        
        # Process the data
        logger.info(f"Processing model: {model}, experiment: {experiment}")
        
        with xr.open_dataset(efp_file) as ds:
            # Extract EFP variables
            efp_vars = [var for var in ds.data_vars if var.startswith('efp_ucomp_div1_QG_')]
            
            if not efp_vars:
                logger.warning(f"No EFP variables found in {efp_file}")
                return False
            
            ds_efp = ds[efp_vars]
            
            # Save the processed data
            output_file = save_path / f'{model}_efp_500hPa.nc'
            ds_efp.to_netcdf(output_file)
            logger.info(f"Successfully saved EFP data for {model} - {experiment} to {output_file}")
            
            # Log some basic info about the saved data
            logger.debug(f"Variables saved: {list(ds_efp.data_vars.keys())}")
            logger.debug(f"Data dimensions: {dict(ds_efp.dims)}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing model {model}: {e}", exc_info=True)
        return False

def main():
    """Main processing function"""
    logger.info("Starting EFP data processing pipeline")
    
    try:
        # Setup paths
        main_path, save_path = setup_paths()
        
        # Get model list
        models = get_model_list(main_path)
        
        # Process each model
        success_count = 0
        for i, model in enumerate(models, 1):
            logger.info(f"Processing model {i}/{len(models)}: {model}")
            
            if process_model(model, main_path, save_path):
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