import xarray as xr
import numpy as np
import pandas as pd
import os
import glob

# ── paths ───────────────────────────────────────────────────────────────────
DATA_ROOT = (
    '/home/users/cturrell/documents/eddy_feedback/'
    'b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015'
)

OUT_FILE = os.path.join(DATA_ROOT, 'b_annual_cycle_cmip6_hist.csv')

# ── constants ────────────────────────────────────────────────────────────────
CENTRAL_MONTH_DICT = {
    'DJF': 7, 'JFM': 8, 'FMA': 9, 'MAM': 10,
    'AMJ': 11, 'MJJ': 12, 'JJA': 1, 'JAS': 2,
    'ASO': 3, 'SON': 4, 'OND': 5, 'NDJ': 6,
}
SEASON_LIST   = list(CENTRAL_MONTH_DICT.keys())
DIV1_VARIANTS = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
HEMISPHERES   = ['n', 's']
VA_STR        = '_va'


def discover_models(data_root):
    """Return sorted list of (model_name, dataset_path) for all available models."""
    models = []
    for model_dir in sorted(glob.glob(os.path.join(data_root, '*'))):
        nc_path = os.path.join(model_dir, '6hrPlevPt', 'b_dataset.nc')
        if os.path.isfile(nc_path):
            models.append((os.path.basename(model_dir), nc_path))
    return models


def extract_b_annual_cycle(b_dataset, model_name):
    """
    Extract all b annual-cycle values from one model dataset.
    Returns a list of dicts with keys: model, variant, hemisphere, season,
    central_month, b.
    """
    rows = []
    for variant in DIV1_VARIANTS:
        for hemisphere in HEMISPHERES:
            for season in SEASON_LIST:
                b_var_name = f'ucomp{VA_STR}_{variant}{VA_STR}_b_{hemisphere}_{season}'
                if b_var_name in b_dataset:
                    b_val = float(b_dataset[b_var_name].mean('lag').values)
                else:
                    b_val = np.nan
                rows.append({
                    'model':         model_name,
                    'variant':       variant,
                    'hemisphere':    hemisphere,
                    'season':        season,
                    'b':             b_val,
                })
    return rows


# ── main ─────────────────────────────────────────────────────────────────────
models = discover_models(DATA_ROOT)
print(f'Found {len(models)} models: {[m for m, _ in models]}')

all_rows = []
for model_name, nc_path in models:
    print(f'  Processing {model_name} ...')
    with xr.open_dataset(nc_path) as ds:
        all_rows.extend(extract_b_annual_cycle(ds, model_name))

df = pd.DataFrame(all_rows)
df.to_csv(OUT_FILE, index=False)
print(f'\nSaved {len(df)} rows → {OUT_FILE}')
print(df.head())
