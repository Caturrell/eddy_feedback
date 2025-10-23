import xarray as xr
import os
import logging
import numpy as np
import dask

import functions.eddy_feedback as ef
import functions.data_wrangling as dw
import functions.aos_functions as aos


# -------------------------------
# Configure logging
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_QG_epfluxes(ds):
    """Return dataset with QG EP fluxes added"""
    ds = dw.data_checker1000(ds)
    ucomp, vcomp, temp = ds.u, ds.v, ds.t

    logging.info("Computing ubar")
    ds['ubar'] = ucomp.mean('lon')

    logging.info("Computing QG EP fluxes")
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp)

    ds['ep1_QG'] = ep1
    ds['ep2_QG'] = ep2
    ds['div1_QG'] = div1
    ds['div2_QG'] = div2
    return ds


def calculate_primitive_epfluxes(ds):
    """Return dataset with primitive EP fluxes added"""
    ds = dw.data_checker1000(ds)
    ucomp, vcomp, temp, omega = ds.u, ds.v, ds.t, ds.omega

    logging.info("Computing ubar")
    ds['ubar'] = ucomp.mean('lon')

    logging.info("Computing PRIMITIVE EP fluxes")
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(
        ucomp, vcomp, temp, w=omega, do_ubar=True, ref='instant'
    )

    ds['ep1_pr'] = ep1
    ds['ep2_pr'] = ep2
    ds['div1_pr'] = div1
    ds['div2_pr'] = div2
    return ds


def load_6h_data_for_year(source_path, year):
    """Load and merge u,v,t,omega 6-hourly data for one year"""
    variables = ['ucomp', 'vcomp', 'temp', 'omega']
    datasets = []
    for var_folder in variables:
        var_path = os.path.join(source_path, var_folder)
        logging.info(f"Loading {year} from {var_path}")
        ds = xr.open_mfdataset(f"{var_path}/{year}*.nc", combine="by_coords")
        ds = dw.data_checker1000(ds)
        datasets.append(ds)
    return xr.merge(datasets)


def process_year(source_path, dest_root, year):
    """Compute EP fluxes (QG and primitive), save 6h + daily in separate dirs"""
    logging.info(f"Processing year {year}")
    ds_base = load_6h_data_for_year(source_path, year)

    # --- QG fluxes ---
    ds_QG = calculate_QG_epfluxes(ds_base.copy())
    path_QG = os.path.join(dest_root, '6h_uvtw')
    os.makedirs(path_QG, exist_ok=True)

    save_6h_QG = os.path.join(path_QG, 'QG_epf_uvtw', f"{year}_6h_uvtw_epf_QG.nc")
    save_daily_QG = os.path.join(path_QG, 'daily_averages', 'QG_epf_uvtw', f"{year}_6h_daily-avg_uvtw_epf_QG.nc")

    if not os.path.exists(save_daily_QG):
        logging.info(f"Saving QG 6-hourly: {save_6h_QG}")
        dask.compute(ds_QG.to_netcdf(save_6h_QG, compute=False))

        logging.info("Resampling QG dataset to daily means")
        ds_QG_daily = ds_QG.resample(time="1D").mean()

        logging.info(f"Saving QG daily: {save_daily_QG}")
        dask.compute(ds_QG_daily.to_netcdf(save_daily_QG, compute=False))
    else:
        logging.info(f"QG daily file exists for {year}, skipping...")

    # --- Primitive fluxes ---
    ds_pr = calculate_primitive_epfluxes(ds_base.copy())
    path_pr = os.path.join(dest_root, '6h_uvtw')
    os.makedirs(path_pr, exist_ok=True)

    save_6h_pr = os.path.join(path_pr, "pr_epf_uvtw", f"{year}_6h_uvtw_epf_pr.nc")
    save_daily_pr = os.path.join(path_pr, 'daily_averages', "pr_epf_uvtw", f"{year}_6h_daily-avg_uvtw_epf_pr.nc")

    if not os.path.exists(save_daily_pr):
        logging.info(f"Saving PRIMITIVE 6-hourly: {save_6h_pr}")
        dask.compute(ds_pr.to_netcdf(save_6h_pr, compute=False))

        logging.info("Resampling PRIMITIVE dataset to daily means")
        ds_pr_daily = ds_pr.resample(time="1D").mean()

        logging.info(f"Saving PRIMITIVE daily: {save_daily_pr}")
        dask.compute(ds_pr_daily.to_netcdf(save_daily_pr, compute=False))
    else:
        logging.info(f"PRIMITIVE daily file exists for {year}, skipping...")


if __name__ == "__main__":

    source_path = "/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/6h_uvtw"
    dest_root = "/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016"

    for year in np.arange(1958, 2017):
        try:
            process_year(source_path, dest_root, year)
        except Exception as e:
            logging.exception(f"Failed processing year {year}: {e}")
            import pdb; pdb.set_trace()

    logging.info("All processing complete.")
