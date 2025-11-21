import xarray as xr
import os
import logging

import functions.eddy_feedback as ef
import functions.data_wrangling as dw
import functions.aos_functions as aos


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Define input/output directories
data_path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/6h_uvtw/QG_epf_uvtw'
save_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily/k123_6h_QG_epfluxes'

# Make sure output directory exists
os.makedirs(save_dir, exist_ok=True)

# List all .nc files
files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.nc')])


def calculate_QG_epfluxes_k123(ds):
    """
    Input: Xarray dataset
            - Variables required: u,v,t
            - Dim labels: (time, lon, lat, level) 
    Output: Xarray dataset with EP fluxes calculated
    """
    logger.info("Running data checker")
    ds = dw.data_checker1000(ds)

    ucomp = ds.u
    vcomp = ds.v
    temp = ds.t
    
    logger.info("Computing EP fluxes for waves 1,2,3")
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp, wave=[1, 2, 3])

    # save variables to dataset
    ds['ep1_QG_123'] = (ep1.dims, ep1.values)
    ds['ep2_QG_123'] = (ep2.dims, ep2.values)
    ds['div1_QG_123'] = (div1.dims, div1.values)
    ds['div2_QG_123'] = (div2.dims, div2.values)

    # ds = ds.load()
    logger.info("EP fluxes added to dataset")
    return ds


if __name__ == "__main__":
    try:
        for fpath in files:
            fname = os.path.basename(fpath)            
            logger.info(f"Processing file: {fname}")

            # Open dataset
            ds = xr.open_dataset(fpath)

            # Compute EP fluxes
            ds = calculate_QG_epfluxes_k123(ds)

            # Decompose >k=3
            for var_name_to_decompose in ['ep1_QG', 'ep2_QG', 'div1_QG', 'div2_QG']:
                var = f'{var_name_to_decompose}'
                logger.info(f"Decomposing {var} > k=3")
                ds[f'{var}_gt3'] = ds[f'{var}'] - ds[f'{var}_123']
                
            logger.info("Resampling to daily means")
            ds = ds.resample(time='1D').mean()

            # Define output file path
            name_only = os.path.splitext(fname)[0]
            save_path = os.path.join(save_dir, f"{name_only}_k123_dm.nc")
            logger.info(f"Saving dataset to: {save_path}")

            # Save processed dataset
            
            ds.to_netcdf(save_path)

            logger.info(f"Finished processing {fname}")

        logger.info("All files processed successfully.")

    except Exception as e:
        logger.exception(f"Script failed with error: {e}")
