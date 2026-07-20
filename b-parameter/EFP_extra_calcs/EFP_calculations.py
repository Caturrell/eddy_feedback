import xarray as xr
import os
import numpy as np
import warnings
import logging
import json

import functions.data_wrangling as dw

logger = logging.getLogger(__name__)

# Level variants to compute EFP for:
#   mean_level        - mean over the 600-200hPa layer (troposphere)
#   500hPa             - single level, nearest to 500hPa
#   250_500_850hPa      - mean over {250, 500, 850}hPa (each selected via nearest)
LEVEL_VARIANTS = ('mean_level', '500hPa', '250_500_850hPa')

SEASON_MONTH_DICT = {
    'DJF': [12, 1, 2], 'JFM': [1, 2, 3], 'FMA': [2, 3, 4], 'MAM': [3, 4, 5],
    'AMJ': [4, 5, 6], 'MJJ': [5, 6, 7], 'JJA': [6, 7, 8], 'JAS': [7, 8, 9],
    'ASO': [8, 9, 10], 'SON': [9, 10, 11], 'OND': [10, 11, 12], 'NDJ': [11, 12, 1],
    'ANN': list(range(1, 13)),
}

CONFIGS = [
    ('efp_nh',      'div1_QG',     False),
    ('efp_nh_123',  'div1_QG_123', False),
    ('efp_nh_gt3',  'div1_QG_gt3', False),
    ('efp_sh',      'div1_QG',     True),
    ('efp_sh_123',  'div1_QG_123', True),
    ('efp_sh_gt3',  'div1_QG_gt3', True),
]


def seasonal_mean(ds, months, cut_ends=False):
    logger.info(f"Computing mean for months: {months}, cut_ends={cut_ends}")

    if not (isinstance(months, list) and all(isinstance(m, int) and 1 <= m <= 12 for m in months)):
        raise ValueError(f"`months` must be a list of integers between 1-12. Got: {months}")
    if len(months) not in (3, 12):
        raise ValueError(f"`months` must have 3 elements (a season) or 12 (annual mean). Got: {months}")

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
    return result.rename({'season_year': 'time'})


def calculate_efp(ds, months, calc_south_hemis, which_div1, time_slice, cut_ends, level_variant):
    hemi = 'Southern' if calc_south_hemis else 'Northern'
    logger.info(
        f"Calculating EFP for {hemi} Hemisphere, div1: {which_div1}, "
        f"level_variant: {level_variant}, months: {months}"
    )

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
    logger.info(f"Means calculated. Dataset shape: {ds.sizes}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = xr.corr(ds[which_div1], ds.ubar, dim='time').load()**2

        corr = corr.sel(lat=efp_lat_slice)

        if level_variant == 'mean_level':
            corr = corr.sel(level=slice(600., 200.)).mean('level')
        elif level_variant == '500hPa':
            corr = corr.sel(level=500., method='nearest')
        elif level_variant == '250_500_850hPa':
            corr = corr.sel(level=[250., 500., 850.], method='nearest').mean('level')
        else:
            raise ValueError(f"Unknown level_variant: {level_variant}")

        weights = np.cos(np.deg2rad(corr.lat))
        efp = corr.weighted(weights).mean('lat')

        efp_value = round(float(efp.values), 4)
        logger.info(f"EFP = {efp_value}")
        return efp_value

    except Exception as e:
        logger.error(f"Error during EFP calculation: {e}")
        raise RuntimeError(f"Error during Eddy Feedback Parameter calculation: {e}")


def compute_and_save_efp(dataset, output_dir, time_slice=None, cut_ends=False):
    logger.info(f"Starting computation of EFPs. output_dir={output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    json_paths = {
        variant: os.path.join(output_dir, f"efp_results_{variant}.json")
        for variant in LEVEL_VARIANTS
    }

    all_results = {}
    for variant, path in json_paths.items():
        if os.path.exists(path):
            logger.info(f"{path} already exists — loading cached results.")
            with open(path) as f:
                all_results[variant] = json.load(f)
        else:
            all_results[variant] = {}

    for key, which_div1, calc_south_hemis in CONFIGS:
        logger.info(f"Processing configuration: {key}")
        for variant in LEVEL_VARIANTS:
            all_results[variant].setdefault(key, {})

        for season, months in SEASON_MONTH_DICT.items():
            logger.info(f"-> Processing {key}, season={season}")

            if all(season in all_results[variant][key] for variant in LEVEL_VARIANTS):
                logger.info(f"   All level variants already exist for {key}-{season}, skipping.")
                continue

            # Use a local variable so the outer dataset is never mutated.
            ds = dataset
            start_year = ds.time.dt.year[0].values
            end_year   = ds.time.dt.year[-1].values
            if season == 'DJF':
                ds = ds.sel(time=slice(f'{start_year}-03', f'{end_year}-11'))
            elif season == 'NDJ':
                ds = ds.sel(time=slice(f'{start_year}-02', f'{end_year}-10'))

            for variant in LEVEL_VARIANTS:
                if season not in all_results[variant][key]:
                    efp_value = calculate_efp(
                        ds, months=months, calc_south_hemis=calc_south_hemis,
                        which_div1=which_div1, time_slice=time_slice,
                        cut_ends=cut_ends, level_variant=variant,
                    )
                    all_results[variant][key][season] = {"efp": efp_value, "months": months}

    for variant, path in json_paths.items():
        with open(path, "w") as f:
            json.dump(all_results[variant], f, indent=2)
        logger.info(f"Saved {variant} EFPs to: {path}")

    logger.info("Completed all configurations.")
    return all_results


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
    logger.info("Dataset loaded and preprocessed.")

    time_slice = slice('1979', '2014')
    cut_ends = True

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '1979_2014')
    logger.info(f"Output directory: {output_dir}")

    compute_and_save_efp(
        ds6h,
        output_dir=output_dir,
        time_slice=time_slice,
        cut_ends=cut_ends,
    )

    logger.info("Script completed.")
