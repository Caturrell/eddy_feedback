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

def seasonal_mean(ds, months, take_mean=False):
    logger.info(f"Computing seasonal mean for months: {months}, take_mean={take_mean}")
    
    # pre-processing checks
    if not (isinstance(months, list) and all(isinstance(m, int) and 1 <= m <= 12 for m in months)):
        raise ValueError(f"`months` must be a list of 3 integers between 1–12. Got: {months}")
    if len(months) != 3:
        raise ValueError(f"`months` must have exactly 3 elements. Got: {months}")

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
    

def calculate_efp_annual_cycle(ds, months, calc_south_hemis=False, which_div1=None):
    hemi = 'Southern' if calc_south_hemis else 'Northern'
    logger.info(f"Calculating EFP annual cycle for {hemi} Hemisphere, div1: {which_div1}, months: {months}")
    
    if calc_south_hemis:
        ds = ds.sel(lat=slice(-90, 0))
        efp_lat_slice = slice(-75, -25)
    else:
        ds = ds.sel(lat=slice(0, 90))
        efp_lat_slice = slice(25, 75)
        
    # calc seasonal means
    ds = seasonal_mean(ds, months=months)
    logger.info(f"Seasonal means calculated. Dataset shape: {ds.sizes}")

    logger.debug("Computing correlation and EFP.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = xr.corr(ds[which_div1], ds.ubar, dim='ens_ax').load()**2

        corr = corr.sel(lat=efp_lat_slice, level=slice(600., 200.))
        corr = corr.mean('level')

        weights = np.cos(np.deg2rad(corr.lat))
        efp = corr.weighted(weights).mean('lat')

        efp_value = round(float(efp.values[0]), 4)
        logger.info(f"EFP = {efp_value}")
        return efp_value
    
    except Exception as e:
        logger.error(f"Error during EFP calculation: {e}")
        raise RuntimeError(f"Error during Eddy Feedback Parameter calculation: {e}")
    

def compute_and_save_efp_seasonal(dataset, output_dir):
    logger.info(f"Starting computation of seasonal EFPs. Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    season_month_dict = {
        'DJF': [12,1,2], 'JFM': [1,2,3], 'FMA': [2,3,4], 'MAM': [3,4,5],
        'JJA': [6,7,8], 'JAS': [7,8,9],
        'ASO': [8,9,10], 'SON': [9,10,11], 'OND': [10,11,12], 'NDJ': [11,12,1]
    }

    configs = [
        ('efp_nh',      'divFy',     False),
        ('efp_nh_123',  'divFy_k123', False),
        ('efp_nh_gt3',  'divFy_gt3', False),
        ('efp_sh',      'divFy',     True),
        ('efp_sh_123',  'divFy_k123', True),
        ('efp_sh_gt3',  'divFy_gt3', True),
    ]

    # dictionary to hold all results
    all_results = {}

    for key, which_div1, calc_south_hemis in configs:
        logger.info(f"Processing configuration: {key}")
        result = {}

        for season, months in season_month_dict.items():
            logger.info(f"-> Computing result for config={key}, season={season}, months={months}")
                
            # Calculate EFP for the current season
            efp = calculate_efp_annual_cycle(
                dataset,
                months=months,
                calc_south_hemis=calc_south_hemis,
                which_div1=which_div1
            )

            result[season] = {
                "efp": efp,
                "months": months
            }

        all_results[key] = result

    # Save all results into a single JSON file
    json_path = os.path.join(output_dir, "efp_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Saved all results to JSON: {json_path}")
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
    
    pamip_path = '/home/links/ct715/data_storage/PAMIP/processed_daily'
    model_extract_loc = os.path.join(pamip_path, 'daily_efp_mon-avg')
    models = os.listdir(model_extract_loc)
    
    datasets = {}
    for model in models:
        
        logger.info(f"Loading dataset for model: {model}")
    
        # Import k123 data
        k123_path = os.path.join(pamip_path, 'k123_daily_efp_mon-avg', model, '*.nc')
        k123 = xr.open_mfdataset(k123_path, combine='nested', concat_dim='ens_ax')
        
        # import daily data
        daily_path = os.path.join(pamip_path, 'daily_efp_mon-avg', model, '*.nc')
        daily = xr.open_mfdataset(daily_path, combine='nested', concat_dim='ens_ax')
        
        # import daily data
        daily['divFy_k123'] = k123.divFy_k123
        daily['divFy_gt3'] = daily.divFy - daily.divFy_k123
        pamip = daily[['ubar', 'divFy', 'divFy_k123', 'divFy_gt3']]
        
        datasets[model] = pamip

    logger.info("Datasets loaded and preprocessed.")

    for idx, (model, dataset) in enumerate(datasets.items()):
        
        logger.info(f"Processing dataset {idx+1}/{len(datasets)}")

        save_dir = f'/home/links/ct715/eddy_feedback/chapter1/annual_cycle/data/PAMIP/{model}'
        os.makedirs(save_dir, exist_ok=True)

        results = compute_and_save_efp_seasonal(dataset, output_dir=save_dir)

    logger.info("Script completed.")

