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
    """
    Compute and save seasonal EFP values (mean-level and 500 hPa) for all
    rolling 3-month seasons.

    Parameters
    ----------
    dataset      : xr.Dataset  – Input dataset with ubar and div1_* variables.
    output_dir   : str         – Directory for mean-level JSON output.
    output_dir_500 : str       – Directory for 500 hPa JSON output.
    time_slice   : slice|None  – Optional temporal sub-selection.
    cut_ends     : bool        – Whether to trim incomplete seasons at the edges.
    dataset_freq : str         – Label for logging ('6hourly' or 'daily').
    """
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
        ('efp_pr_nh', 'div1_pr', False),
        ('efp_QG_nh', 'div1_QG', False),
        ('efp_pr_sh', 'div1_pr', True),
        ('efp_QG_sh', 'div1_QG', True),
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
        if key not in all_results:
            all_results[key] = {}
        if key not in all_results_500:
            all_results_500[key] = {}

        for season, months in season_month_dict.items():
            logger.info(f"-> Processing {key}, season={season}")

            if season in all_results[key] and season in all_results_500[key]:
                logger.info(f"   Both mean and 500 hPa EFPs already exist for {key}-{season}, skipping.")
                continue

            # Trim the edges of cross-year seasons to avoid partial seasons.
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Script started.")

    path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/processed_efp'
    logger.info(f"Loading datasets from: {path}")

    ds6h = xr.open_mfdataset(os.path.join(path, '6h_*.nc'))
    ds6h = dw.data_checker1000(ds6h, check_vars=False)

    day = xr.open_mfdataset(os.path.join(path, 'daily_*.nc'))
    day = dw.data_checker1000(day, check_vars=False)

    logger.info("Datasets loaded and preprocessed.")

    CUT_ENDS = True
    if CUT_ENDS:
        logger.info("CUT_ENDS=True: incomplete seasons at dataset edges will be trimmed.")
        data_dir = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/various_efp_methods/data/efp_cut_ends'
    else:
        data_dir = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/various_efp_methods/data'

    os.makedirs(data_dir, exist_ok=True)

    time_slices = [slice('1958', '2016'), slice('1979', '2016')]

    datasets = [(ds6h, '6hourly_efp', '6hourly'), (day, 'daily_efp', 'daily')]

    for idx, (dataset, folder, freq) in enumerate(datasets):
        logger.info(f"Processing dataset {idx + 1}/{len(datasets)}: {freq}")

        for time_slice in time_slices:
            ts_str = f"{time_slice.start}_{time_slice.stop}".replace('-', '')
            logger.info(f"Processing time slice: {time_slice.start} to {time_slice.stop} ({ts_str})")

            # Both output dirs share the same data_dir root, so CUT_ENDS
            # branching above applies automatically to both paths.
            save_dir     = os.path.join(data_dir, folder,                  ts_str)
            save_dir_500 = os.path.join(data_dir, f"500hPa_{freq}_efp",    ts_str)

            os.makedirs(save_dir,     exist_ok=True)
            os.makedirs(save_dir_500, exist_ok=True)

            logger.info(f"  Mean-level output : {save_dir}")
            logger.info(f"  500 hPa output    : {save_dir_500}")

            compute_and_save_efp_seasonal(
                dataset,
                output_dir=save_dir,
                output_dir_500=save_dir_500,
                time_slice=time_slice,
                cut_ends=CUT_ENDS,
                dataset_freq=freq,
            )

    logger.info("Script completed.")