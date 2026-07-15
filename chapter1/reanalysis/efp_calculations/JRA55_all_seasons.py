"""
    !!!!!!!!!!!!!!
    !!!!!!!!!!!!!! NOT USED YET !!!!!
    !!!!!!!!!!!!!!

Calculate the Eddy Feedback Parameter (EFP) for JRA55 reanalysis over all
12 three-month "seasons" (djf, jfm, fma, ..., ndj), for both hemispheres.

Follows the same methodology as JRA55_EFP.ipynb (ef.calculate_efp), but
loops over every rolling 3-month season instead of just the standard
NH-winter (djf) / SH-winter (jas) seasons.

Edit the CONFIG block below to change time frequency, period, div1 method,
level method (500hPa slice vs full-level), or which seasons/hemispheres to
compute.
"""
import os

import pandas as pd
import xarray as xr

import functions.eddy_feedback as ef

# =====================================================================
# CONFIG - edit here to change what gets calculated / saved
# =====================================================================

DATA_PATH = '/home/links/ct715/data_storage/reanalysis/jra55_daily/processed_efp'
OUTPUT_CSV = '/home/links/ct715/eddy_feedback/chapter1/reanalysis/data/jra55_efp_all_seasons.csv'

FILENAME_MAP = {
    '6h': '6h_ubar_epf-pr-QG_1MS_1958-2016.nc',
    'daily': 'daily_ubar_epf-pr-QG_1MS_1958-2016.nc',
}

TIME_FREQ = '6h'               # '6h' or 'daily'
PERIOD = ('1979', '2014')      # (start_year, end_year), inclusive
DIV1_METHOD = 'div1_QG'        # 'div1_pr' or 'div1_QG'
SLICE_500HPA = True            # True: 500hPa slice, False: full-level (600-200hPa mean)

SEASONS = ['djf', 'jfm', 'fma', 'mam', 'amj', 'mjj',
           'jja', 'jas', 'aso', 'son', 'ond', 'ndj']
HEMISPHERES = [False, True]    # calc_south_hemis: False -> NH, True -> SH

# =====================================================================


def load_dataset(time_freq):
    path = os.path.join(DATA_PATH, FILENAME_MAP[time_freq])
    return xr.open_dataset(path)


def calculate_all_seasons(ds, time_freq, period, div1_method, slice_500hPa,
                           seasons, hemispheres):
    """
    Calculate EFP for every combination of season/hemisphere and return
    the results as a long-format DataFrame.
    """
    ds_period = ds.sel(time=slice(f"{period[0]}-01", f"{period[1]}-12"))

    results = []
    for calc_south_hemis in hemispheres:
        hemisphere = 'SH' if calc_south_hemis else 'NH'
        for season in seasons:
            efp = ef.calculate_efp(
                ds_period,
                which_div1=div1_method,
                data_type='reanalysis',
                calc_south_hemis=calc_south_hemis,
                slice_500hPa=slice_500hPa,
                season=season,
            )
            results.append({
                'season': season,
                'hemisphere': hemisphere,
                'div1_method': div1_method,
                'level_method': '500hPa' if slice_500hPa else 'full_level',
                'time_freq': time_freq,
                'period': f"{period[0]}-{period[1]}",
                'efp': efp,
            })

    return pd.DataFrame(results)


if __name__ == '__main__':
    ds = load_dataset(TIME_FREQ)

    df_efp = calculate_all_seasons(
        ds, TIME_FREQ, PERIOD, DIV1_METHOD, SLICE_500HPA, SEASONS, HEMISPHERES
    )
    print(df_efp)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_efp.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
