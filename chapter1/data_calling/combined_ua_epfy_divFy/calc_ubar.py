#!/usr/bin/env python3

import xarray as xr
from pathlib import Path
import logging
import tempfile
import shutil

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def main():
    # Setup logger
    setup_logger()
  
    # Define directory path
    base_dir = Path('/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC')

    # Get list of model files
    model_files = sorted(base_dir.glob("*_1.1_u_ubar_epfy_divFy.nc"))

    for file_path in model_files:
        model = file_path.name.split('_')[0]

        try:
            ds_model = xr.open_dataset(file_path)

            # Check for 'ubar', compute if missing
            if 'ubar' not in ds_model.variables:
                if 'u' in ds_model.variables:
                    logging.info(f"{model}: 'ubar' not found. Calculating zonal mean of 'u'.")
                    ds_model['ubar'] = ds_model['u'].mean(dim='lon')

                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
                        temp_path = Path(tmp.name)
                    ds_model.to_netcdf(temp_path)

                    # Replace original file
                    shutil.move(str(temp_path), file_path)
                    logging.info(f"{model}: Updated dataset saved with 'ubar'.")
                else:
                    logging.warning(f"{model}: 'u' not found. Cannot compute 'ubar'.")
            else:
                logging.info(f"{model}: 'ubar' already exists.")

            ds_model.close()

        except Exception as e:
            logging.error(f"Error processing {model}: {e}")

if __name__ == "__main__":
    main()
