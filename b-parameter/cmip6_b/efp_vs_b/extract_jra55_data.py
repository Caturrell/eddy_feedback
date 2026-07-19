"""
Extract JRA55 (1979-2014) EFP and b annual-cycle values into the same
long-format CSV shape as the CMIP6 csvs used by efp_vs_b_plot.py
(columns: model, variant, hemisphere, season, efp / b).

Sources:
  b   -> b_dataset.nc from jra55_250-850_calc_efp_b.py, level_250_500_850hPa
         config (pressure-weighted 250/500/850hPa EOFs) -- the JRA55 analogue
         of CMIP6's 250-500-850hPa_dm method.
  efp -> efp_results_500hPa.json from annual_cycles_QG_k.py, cut_ends=False,
         time_slice=1979-2014 -- matches the 500hPa methodology used to
         build efp_annual_cycle_cmip6_hist.csv.
"""
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

MODEL_NAME = 'JRA55'

B_NC_PATH = (
    '/home/links/ct715/eddy_feedback/b-parameter/b_methodology/all_plots_true/'
    'jra55_850_sit_plots/1979_2014/6hourly/level_250_500_850hPa/b_dataset.nc'
)
EFP_JSON_PATH = (
    '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/'
    'data/500hPa_6hourly_efp/1979_2014/efp_results_500hPa.json'
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
B_OUT_FILE = os.path.join(OUT_DIR, 'jra55_b_annual_cycle.csv')
EFP_OUT_FILE = os.path.join(OUT_DIR, 'jra55_efp_annual_cycle.csv')
B_ALL_TIME_OUT_FILE = os.path.join(OUT_DIR, 'jra55_b_all_time.csv')
B_NATIVE_OUT_FILE = os.path.join(OUT_DIR, 'jra55_b_native.csv')

DIV1_VARIANTS = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
HEMISPHERES = ['n', 's']
SEASON_LIST = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
               'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
VA_STR = '_va'

EFP_KEY_MAP = {
    'efp_nh':     ('n', 'div1_QG'),
    'efp_nh_123': ('n', 'div1_QG_123'),
    'efp_nh_gt3': ('n', 'div1_QG_gt3'),
    'efp_sh':     ('s', 'div1_QG'),
    'efp_sh_123': ('s', 'div1_QG_123'),
    'efp_sh_gt3': ('s', 'div1_QG_gt3'),
}


def extract_b():
    rows = []
    with xr.open_dataset(B_NC_PATH) as ds:
        for variant in DIV1_VARIANTS:
            for hemisphere in HEMISPHERES:
                for season in SEASON_LIST:
                    b_var_name = f'ucomp{VA_STR}_{variant}{VA_STR}_b_{hemisphere}_{season}'
                    if b_var_name in ds:
                        b_val = float(ds[b_var_name].mean('lag').values)
                    else:
                        b_val = np.nan
                    rows.append({
                        'model': MODEL_NAME,
                        'variant': variant,
                        'hemisphere': hemisphere,
                        'season': season,
                        'b': b_val,
                    })
    return pd.DataFrame(rows)


def extract_b_all_time():
    rows = []
    with xr.open_dataset(B_NC_PATH) as ds:
        for variant in DIV1_VARIANTS:
            for hemisphere in HEMISPHERES:
                b_var_name = f'ucomp{VA_STR}_{variant}{VA_STR}_b_{hemisphere}_all_time'
                if b_var_name in ds:
                    b_val = float(ds[b_var_name].mean('lag').values)
                else:
                    b_val = np.nan
                rows.append({
                    'model': MODEL_NAME,
                    'variant': variant,
                    'hemisphere': hemisphere,
                    'season': 'all_time',
                    'b': b_val,
                })
    return pd.DataFrame(rows)


def extract_b_native():
    """Same as extract_b()/extract_b_all_time() but reads the native (per-level,
    non vertically-averaged) EOF b variables, i.e. without the '_va' suffix."""
    rows = []
    with xr.open_dataset(B_NC_PATH) as ds:
        for variant in DIV1_VARIANTS:
            for hemisphere in HEMISPHERES:
                for season in SEASON_LIST + ['all_time']:
                    b_var_name = f'ucomp_{variant}_b_{hemisphere}_{season}'
                    if b_var_name in ds:
                        b_val = float(ds[b_var_name].mean('lag').values)
                    else:
                        b_val = np.nan
                    rows.append({
                        'model': MODEL_NAME,
                        'variant': variant,
                        'hemisphere': hemisphere,
                        'season': season,
                        'b': b_val,
                    })
    return pd.DataFrame(rows)


def extract_efp():
    with open(EFP_JSON_PATH) as f:
        data = json.load(f)

    rows = []
    for key, (hemisphere, variant) in EFP_KEY_MAP.items():
        for season, info in data[key].items():
            rows.append({
                'model': MODEL_NAME,
                'variant': variant,
                'hemisphere': hemisphere,
                'season': season,
                'efp': info['efp'],
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df_b = extract_b()
    df_b.to_csv(B_OUT_FILE, index=False)
    print(f'Saved {len(df_b)} rows -> {B_OUT_FILE}')

    df_efp = extract_efp()
    df_efp.to_csv(EFP_OUT_FILE, index=False)
    print(f'Saved {len(df_efp)} rows -> {EFP_OUT_FILE}')

    df_b_all_time = extract_b_all_time()
    df_b_all_time.to_csv(B_ALL_TIME_OUT_FILE, index=False)
    print(f'Saved {len(df_b_all_time)} rows -> {B_ALL_TIME_OUT_FILE}')

    df_b_native = extract_b_native()
    df_b_native.to_csv(B_NATIVE_OUT_FILE, index=False)
    print(f'Saved {len(df_b_native)} rows -> {B_NATIVE_OUT_FILE}')
