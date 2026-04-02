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
        year  = time.dt.year
        month = time.dt.month
        if months[0] > months[-1]:
            # Cross-year season — two cases:
            if months[0] == 12:
                # DJF-type [12,1,2]: December belongs to the *next* year's season label.
                year = xr.where(month == 12, year + 1, year)
            else:
                # NDJ-type [11,12,1]: early-calendar months (those < months[0]) are
                # the tail of the previous year's season, so assign year - 1.
                early_months = [m for m in months if m < months[0]]
                year = xr.where(month.isin(early_months), year - 1, year)
        return year

    season_year = assign_season_year(ds_season['time'])
    ds_season = ds_season.assign_coords(season_year=('time', season_year.data))
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
    

def compute_and_save_efp_seasonal(dataset, output_dir, output_dir_500, time_slice=None, cut_ends=False, dataset_freq='6hourly'):
    logger.info(
        f"Starting computation of seasonal EFPs. "
        f"output_dir={output_dir}, output_dir_500={output_dir_500}, dataset_freq={dataset_freq}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_500, exist_ok=True)

    season_month_dict = {
        'DJF': [12, 1, 2], 'JFM': [1, 2, 3], 'FMA': [2, 3, 4], 'MAM': [3, 4, 5],
        'AMJ': [4, 5, 6], 'MJJ': [5, 6, 7], 'JJA': [6, 7, 8], 'JAS': [7, 8, 9],
        'ASO': [8, 9, 10], 'SON': [9, 10, 11], 'OND': [10, 11, 12], 'NDJ': [11, 12, 1],
    }

    configs = [
        ('efp_nh',      'div1_QG',     False),
        ('efp_nh_123',  'div1_QG_123', False),
        ('efp_nh_gt3',  'div1_QG_gt3', False),
        ('efp_sh',      'div1_QG',     True),
        ('efp_sh_123',  'div1_QG_123', True),
        ('efp_sh_gt3',  'div1_QG_gt3', True),
    ]

    json_mean = os.path.join(output_dir,     "efp_results.json")
    json_500  = os.path.join(output_dir_500, "efp_results_500hPa.json")

    if os.path.exists(json_mean):
        logger.info(f"{json_mean} already exists — loading cached mean-level results.")
        with open(json_mean) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if os.path.exists(json_500):
        logger.info(f"{json_500} already exists — loading cached 500 hPa results.")
        with open(json_500) as f:
            all_results_500 = json.load(f)
    else:
        all_results_500 = {}

    for key, which_div1, calc_south_hemis in configs:
        logger.info(f"Processing configuration: {key}")
        if key not in all_results:
            all_results[key] = {}
        if key not in all_results_500:
            all_results_500[key] = {}

        for season, months in season_month_dict.items():
            logger.info(f"-> Processing {key}, season={season}")

            if season in all_results[key] and season in all_results_500[key]:
                logger.info(f"   Both mean and 500 hPa EFPs already exist for {key}-{season}, skipping.")
                continue

            # Use a local variable so the outer dataset is never mutated.
            ds = dataset
            start_year = ds.time.dt.year[0].values
            end_year   = ds.time.dt.year[-1].values
            if season == 'DJF':
                ds = ds.sel(time=slice(f'{start_year}-03', f'{end_year}-11'))
            elif season == 'NDJ':
                ds = ds.sel(time=slice(f'{start_year}-02', f'{end_year}-10'))

            if season not in all_results[key]:
                efp_mean = calculate_efp_annual_cycle(
                    ds, months=months, calc_south_hemis=calc_south_hemis,
                    which_div1=which_div1, time_slice=time_slice,
                    cut_ends=cut_ends, slice_500hPa=False,
                )
                all_results[key][season] = {"efp": efp_mean, "months": months}

            if season not in all_results_500[key]:
                efp_500 = calculate_efp_annual_cycle(
                    ds, months=months, calc_south_hemis=calc_south_hemis,
                    which_div1=which_div1, time_slice=time_slice,
                    cut_ends=cut_ends, slice_500hPa=True,
                )
                all_results_500[key][season] = {"efp": efp_500, "months": months}

    with open(json_mean, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(json_500, "w") as f:
        json.dump(all_results_500, f, indent=2)

    logger.info(f"Saved mean-level EFPs to : {json_mean}")
    logger.info(f"Saved 500 hPa EFPs to    : {json_500}")
    logger.info("Completed all configurations.")
    return all_results, all_results_500



if __name__ == "__main__":
    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.info("Script started.")
    
    
    # import 6h data
    path_6h = '/home/links/ct715/data_storage/reanalysis/jra55_daily/k123_6h_QG_epfluxes'
    data_path_6h = os.path.join(path_6h, '*_6h_uvtw_epf_QG_k123_dm.nc')
    logger.info(f"Loading dataset from: {data_path_6h}")
    ds6h = xr.open_mfdataset(data_path_6h)
    ds6h = ds6h[['ubar', 'div1_QG', 'div1_QG_123', 'div1_QG_gt3']]
    ds6h = dw.data_checker1000(ds6h, check_vars=False)
    
    # import daily data
    daily_path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/k123_daily_QG_epfluxes'
    data_path_day = os.path.join(daily_path, '*_daily_averages.nc')
    logger.info(f"Loading dataset from: {data_path_day}")
    day = xr.open_mfdataset(data_path_day)
    day = day[['ubar', 'div1_QG', 'div1_QG_123', 'div1_QG_gt3']]
    day = dw.data_checker1000(day, check_vars=False)
    
    datasets = [(ds6h, '6hourly_efp', '6hourly'), (day, 'daily_efp', 'daily')]

    logger.info("Datasets loaded and preprocessed.")

    BASE = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/data'
    cut_ends_data_dirs = {
        True:  os.path.join(BASE, 'efp_cut_ends'),
        False: BASE,
    }

    time_slices = [slice('1958', '2016'), slice('1979', '2016')]

    for cut_ends, data_dir in cut_ends_data_dirs.items():
        logger.info(f"Processing cut_ends={cut_ends}, data_dir={data_dir}")
        os.makedirs(data_dir, exist_ok=True)

        for idx, (dataset, folder, freq) in enumerate(datasets):
            logger.info(f"Processing dataset {idx + 1}/{len(datasets)}: {freq}")

            for time_slice in time_slices:
                ts_str = f"{time_slice.start}_{time_slice.stop}".replace('-', '')
                logger.info(f"Processing time slice: {time_slice.start} to {time_slice.stop} ({ts_str})")

                save_dir     = os.path.join(data_dir, folder,               ts_str)
                save_dir_500 = os.path.join(data_dir, f"500hPa_{freq}_efp", ts_str)

                os.makedirs(save_dir,     exist_ok=True)
                os.makedirs(save_dir_500, exist_ok=True)

                logger.info(f"  Mean-level output : {save_dir}")
                logger.info(f"  500 hPa output    : {save_dir_500}")

                compute_and_save_efp_seasonal(
                    dataset,
                    output_dir=save_dir,
                    output_dir_500=save_dir_500,
                    time_slice=time_slice,
                    cut_ends=cut_ends,
                    dataset_freq=freq,
                )

    logger.info("Script completed.")

