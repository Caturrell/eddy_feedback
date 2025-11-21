import xarray as xr
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import logging
import json

import functions.eddy_feedback as ef
import functions.data_wrangling as dw
import functions.aos_functions as aos

logger = logging.getLogger(__name__)

def seasonal_mean(ds, months, cut_ends=False, take_mean=False):
    logger.info(f"Computing seasonal mean for months: {months}, cut_ends={cut_ends}, take_mean={take_mean}")
    
    if not (isinstance(months, list) and all(isinstance(m, int) and 1 <= m <= 12 for m in months)):
        raise ValueError(f"`months` must be a list of 3 integers between 1–12. Got: {months}")
    if len(months) != 3:
        raise ValueError(f"`months` must have exactly 3 elements. Got: {months}")

    if cut_ends:
        logger.debug("Cutting incomplete ends to ensure full seasons.")
        first_valid_time = ds['time'].sel(time=ds['time'].dt.month.isin([months[0]])).isel(time=0).values
        last_valid_time  = ds['time'].sel(time=ds['time'].dt.month.isin([months[-1]])).isel(time=-1).values
        ds = ds.sel(time=slice(first_valid_time, last_valid_time))

    ds_season = ds.sel(time=ds['time'].dt.month.isin(months))

    def assign_season_year(time):
        year = time.dt.year
        month = time.dt.month
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
    

def calculate_efp_annual_cycle(ds, months, calc_south_hemis=False, which_div1=None, time_slice=None, cut_ends=False, slice_500hPa=False):
    hemi = 'Southern' if calc_south_hemis else 'Northern'
    mode = "500hPa" if slice_500hPa else "mean-level"
    logger.info(f"Calculating EFP annual cycle for {hemi} Hemisphere, div1: {which_div1}, mode: {mode}, months: {months}")
    
    if calc_south_hemis:
        ds = ds.sel(lat=slice(-90, 0))
        efp_lat_slice = slice(-75, -25)
    else:
        ds = ds.sel(lat=slice(0, 90))
        efp_lat_slice = slice(25, 75)

    if time_slice is not None:
        logger.debug(f"Applying time slice: {time_slice}")
        ds = ds.sel(time=time_slice)
    else:
        logger.info(f"Using full time period: {ds.time.min().values} to {ds.time.max().values}")
        
    ds = seasonal_mean(ds, months=months, cut_ends=cut_ends)
    logger.info(f"Seasonal means calculated. Dataset shape: {ds.sizes}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = xr.corr(ds[which_div1], ds.ubar, dim='time').load()**2

        if slice_500hPa:
            corr = corr.sel(lat=efp_lat_slice)
            corr = corr.sel(level=500., method='nearest')
        else:
            corr = corr.sel(lat=efp_lat_slice, level=slice(600., 200.))
            corr = corr.mean('level')

        weights = np.cos(np.deg2rad(corr.lat))
        efp = corr.weighted(weights).mean('lat')

        efp_value = round(float(efp.values), 4)
        logger.info(f"EFP = {efp_value}")
        return efp_value
    
    except Exception as e:
        logger.error(f"Error during EFP calculation: {e}")
        raise RuntimeError(f"Error during Eddy Feedback Parameter calculation: {e}")
    

def compute_and_save_efp_seasonal(dataset, output_dir, time_slice=None):
    logger.info(f"Starting computation of seasonal EFPs. Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # dedicated 500 hPa output directory
    output_dir_500 = "/home/links/ct715/eddy_feedback/chapter1/annual_cycle/various_efp_methods/data/500hPa_efp"
    os.makedirs(output_dir_500, exist_ok=True)

    season_month_dict = {
        'DJF': [12,1,2], 'JFM': [1,2,3], 'FMA': [2,3,4], 'MAM': [3,4,5],
        'AMJ': [4,5,6], 'MJJ': [5,6,7], 'JJA': [6,7,8], 'JAS': [7,8,9],
        'ASO': [8,9,10], 'SON': [9,10,11], 'OND': [10,11,12], 'NDJ': [11,12,1]
    }

    configs = [
        ('efp_pr_nh', 'div1_pr', False),
        ('efp_QG_nh', 'div1_QG', False),
        ('efp_pr_sh', 'div1_pr', True),
        ('efp_QG_sh', 'div1_QG', True),
    ]

    # Mean-level JSON (saved per dataset/time slice)
    json_mean = os.path.join(output_dir, "efp_results.json")
    # 500 hPa JSON (saved to global directory)
    # use dataset/time_slice info for unique filename
    slice_name = "full" if time_slice is None else f"{time_slice.start}_{time_slice.stop}".replace('-', '')
    json_500 = os.path.join(output_dir_500, f"efp_results_500hPa_{slice_name}.json")

    if os.path.exists(json_mean):
        logger.info(f"{json_mean} already exists — skipping mean-level calculation.")
        with open(json_mean) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if os.path.exists(json_500):
        logger.info(f"{json_500} already exists — skipping 500 hPa calculation.")
        with open(json_500) as f:
            all_results_500 = json.load(f)
    else:
        all_results_500 = {}

    for key, which_div1, calc_south_hemis in configs:
        if key not in all_results:
            all_results[key] = {}
        if key not in all_results_500:
            all_results_500[key] = {}

        for season, months in season_month_dict.items():
            logger.info(f"-> Processing {key}, season={season}")

            # skip if both already done
            if season in all_results[key] and season in all_results_500[key]:
                logger.info(f"   Both mean and 500hPa EFPs already exist for {key}-{season}, skipping.")
                continue

            if season == 'DJF':
                start_year = dataset.time.dt.year[0].values
                end_year = dataset.time.dt.year[-1].values
                dataset = dataset.sel(time=slice(f'{start_year}-03', f'{end_year}-11'))
            elif season == 'NDJ':
                start_year = dataset.time.dt.year[0].values
                end_year = dataset.time.dt.year[-1].values
                dataset = dataset.sel(time=slice(f'{start_year}-02', f'{end_year}-10'))

            # mean-level
            if season not in all_results[key]:
                efp_mean = calculate_efp_annual_cycle(
                    dataset, months=months, calc_south_hemis=calc_south_hemis,
                    which_div1=which_div1, time_slice=time_slice, cut_ends=False, slice_500hPa=False
                )
                all_results[key][season] = {"efp": efp_mean, "months": months}

            # 500 hPa
            if season not in all_results_500[key]:
                efp_500 = calculate_efp_annual_cycle(
                    dataset, months=months, calc_south_hemis=calc_south_hemis,
                    which_div1=which_div1, time_slice=time_slice, cut_ends=False, slice_500hPa=True
                )
                all_results_500[key][season] = {"efp": efp_500, "months": months}

    with open(json_mean, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(json_500, "w") as f:
        json.dump(all_results_500, f, indent=2)

    logger.info(f"Saved mean-level EFPs to: {json_mean}")
    logger.info(f"Saved 500 hPa EFPs to: {json_500}")
    logger.info("Completed all configurations.")
    return all_results, all_results_500


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.info("Script started.")
    
    path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/various_efp_methods'
    logger.info(f"Loading dataset from: {path}")
    
    data_path = os.path.join(path, '6h_*.nc')
    ds6h = xr.open_mfdataset(data_path)
    ds6h = dw.data_checker1000(ds6h, check_vars=False)
    
    data_path = os.path.join(path, 'daily_*.nc')
    day = xr.open_mfdataset(data_path)
    day = dw.data_checker1000(day, check_vars=False)
    
    datasets = [ds6h]  # add day if needed

    logger.info("Datasets loaded and preprocessed.")

    time_slices = [slice('1958', '2016'), slice('1979', '2016')]

    for idx, dataset in enumerate(datasets):
        folder = 'daily_efp' if dataset is day else '6hourly_efp'
        logger.info(f"Processing dataset {idx+1}/{len(datasets)}")

        for time_slice in time_slices:
            logger.info(f"Processing time slice: {time_slice.start} to {time_slice.stop}")
            ts_str = f"{time_slice.start}_{time_slice.stop}".replace('-', '')
            save_dir = f'/home/links/ct715/eddy_feedback/chapter1/annual_cycle/various_efp_methods/data/{folder}/{ts_str}'
            os.makedirs(save_dir, exist_ok=True)
            compute_and_save_efp_seasonal(dataset, output_dir=save_dir, time_slice=time_slice)

    logger.info("Script completed.")
