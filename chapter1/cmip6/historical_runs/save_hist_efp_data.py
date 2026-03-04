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

def process_model(model, main_path, base_save_path, year_range=None):
    """Process EFP data for a single model, handling multiple experiments"""
    model_path = main_path / model
    
    try:
        # Get all experiments for this model
        experiments = sorted(os.listdir(model_path))
        logger.info(f"Found {len(experiments)} experiment(s) for model {model}: {experiments}")
        
        success_count = 0
        
        if len(experiments) > 1:
            raise ValueError(f"Multiple experiments found for model {model}: {experiments}. Please ensure only one experiment per model.")
        
        # Loop over all experiments
        for experiment in experiments:
            try:
                # Determine the year range to use
                if year_range is not None:
                    logger.info(f"Using specified year range: {year_range[0]} - {year_range[1]}")
                    output_experiment_length = f'{year_range[0]}_{year_range[1]}'
                    year_start, year_end = year_range[0], year_range[1]
                else:
                    # Use default based on experiment name
                    if experiment in ['1850_2014', '1850_2015']:
                        output_experiment_length = '1850_2014'
                    elif experiment in ['1950_2014', '1950_2015']:
                        output_experiment_length = '1950_2014'
                    else:
                        logger.warning(f"Unexpected experiment name format: {experiment}, using raw name as length")
                        output_experiment_length = experiment
                    
                    years = output_experiment_length.split('_')
                    year_start, year_end = years[0], years[1]
                
                # Create the appropriate subdirectory
                save_path = base_save_path / output_experiment_length / '6h_efp'
                save_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using output directory: {save_path}")
            
                output_file = save_path / f'{model}_{output_experiment_length}_efp_500hPa.nc'
                if output_file.exists():
                    logger.info(f"Output file already exists, skipping: {output_file}")
                    continue
                
                # Build input filename using the year range
                efp_file_name = f'efp_500hPa_{year_start}-{year_end}.nc'
                efp_file = model_path / experiment / '6hrPlevPt' / efp_file_name
                
                if not efp_file.exists():
                    logger.warning(f"File not found: {efp_file}")
                    continue
                
                # Process the data
                logger.info(f"Processing model: {model}, experiment: {experiment} (using years: {year_start}-{year_end})")
                
                with xr.open_dataset(efp_file) as ds:
                    # Extract EFP variables
                    efp_vars = [var for var in ds.data_vars if var.startswith('efp_ucomp_div1_QG_')]
                    
                    if not efp_vars:
                        logger.warning(f"No EFP variables found in {efp_file}")
                        continue
                    
                    ds_efp = ds[efp_vars]
                    
                    # Save the processed data
                    ds_efp.to_netcdf(output_file)
                    logger.info(f"Successfully saved EFP data for {model} to {output_file}")
                    
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
    
    EFP_RANGE = (1958, 2014)  # Default year range to use for EFP data
    
    try:
        # Setup paths
        main_path, base_save_path = setup_paths()
        
        # Get model list
        models = get_model_list(main_path)
        
        # Process each model
        success_count = 0
        for i, model in enumerate(models, 1):
            logger.info(f"Processing model {i}/{len(models)}: {model}")
            
            if process_model(model, main_path, base_save_path, year_range=EFP_RANGE):
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