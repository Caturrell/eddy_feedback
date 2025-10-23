import xarray as xr
import os
import logging

import functions.eddy_feedback as ef
import functions.data_wrangling as dw
import functions.aos_functions as aos


# Set up logging
logfile = 'calculate_epfluxes.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


daily_path = '/home/links/ct715/data_storage/reanalysis/jra55_daily'
data_path = os.path.join(daily_path, 'jra55_uvtw.nc')


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
    
    logger.info("Computing ubar")
    ds['ubar'] = ds.u.mean('lon')
    
    logger.info("Computing EP fluxes")
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp)

    # save variables to dataset
    ds['ep1_QG'] = (ep1.dims, ep1.values)
    ds['ep2_QG'] = (ep2.dims, ep2.values)
    ds['div1_QG'] = (div1.dims, div1.values)
    ds['div2_QG'] = (div2.dims, div2.values)

    logger.info("Computing EP fluxes for waves 1,2,3")
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp, wave=[1, 2, 3])

    # save variables to dataset
    ds['ep1_QG_123'] = (ep1.dims, ep1.values)
    ds['ep2_QG_123'] = (ep2.dims, ep2.values)
    ds['div1_QG_123'] = (div1.dims, div1.values)
    ds['div2_QG_123'] = (div2.dims, div2.values)

    ds = ds.load()
    logger.info("EP fluxes added to dataset")
    return ds


if __name__ == "__main__":
    try:
        logger.info(f"Opening dataset: {data_path}")
        day = xr.open_mfdataset(data_path, chunks={'time': 30})

        day = calculate_QG_epfluxes_k123(day)

        for var_name_to_decompose in ['ep1_QG', 'ep2_QG', 'div1_QG', 'div2_QG']:
            var = f'{var_name_to_decompose}'
            logger.info(f"Decomposing {var} > k=3")
            day[f'{var}_gt3'] = day[f'{var}'] - day[f'{var}_123']

        save_dpath = os.path.join(daily_path, 'jra55_uvtw_ubar_ep-QG_k.nc')
        logger.info(f"Saving dataset to: {save_dpath}")
        day.to_netcdf(save_dpath)
        logger.info("Script completed successfully")

    except Exception as e:
        logger.exception(f"Script failed with error: {e}")
