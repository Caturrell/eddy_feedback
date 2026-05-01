import json
import numpy as np
import pandas as pd
import os
import glob

# ── paths ───────────────────────────────────────────────────────────────────
DATA_ROOT = (
    '/home/links/ct715/eddy_feedback/chapter1/cmip6/historical_runs/data/1979_2014/6h'
)

OUT_FILE = os.path.join(DATA_ROOT, 'efp_annual_cycle_cmip6_hist.csv')

# ── constants ────────────────────────────────────────────────────────────────
# Maps JSON key → (variant, hemisphere)
EFP_KEY_MAP = {
    'efp_nh':     ('div1_QG',     'n'),
    'efp_nh_123': ('div1_QG_123', 'n'),
    'efp_nh_gt3': ('div1_QG_gt3', 'n'),
    'efp_sh':     ('div1_QG',     's'),
    'efp_sh_123': ('div1_QG_123', 's'),
    'efp_sh_gt3': ('div1_QG_gt3', 's'),
}


def discover_models(data_root):
    """Return sorted list of (model_name, json_path) for all available models."""
    models = []
    for model_dir in sorted(glob.glob(os.path.join(data_root, '*'))):
        model_name = os.path.basename(model_dir)
        json_path = os.path.join(
            model_dir, f'{model_name}_efp_1979_2014_CMIP6_hist_6h.json'
        )
        if os.path.isfile(json_path):
            models.append((model_name, json_path))
    return models


def extract_efp_annual_cycle(efp_data, model_name):
    """
    Extract all EFP annual-cycle values from one model's JSON data.
    Returns a list of dicts with keys: model, variant, hemisphere, season, efp.
    """
    rows = []
    for json_key, (variant, hemisphere) in EFP_KEY_MAP.items():
        season_dict = efp_data.get(json_key, {})
        for season, values in season_dict.items():
            efp_val = values['efp'] if isinstance(values, dict) else np.nan
            rows.append({
                'model':      model_name,
                'variant':    variant,
                'hemisphere': hemisphere,
                'season':     season,
                'efp':        efp_val,
            })
    return rows


# ── main ─────────────────────────────────────────────────────────────────────
models = discover_models(DATA_ROOT)
print(f'Found {len(models)} models: {[m for m, _ in models]}')

all_rows = []
for model_name, json_path in models:
    print(f'  Processing {model_name} ...')
    with open(json_path) as f:
        efp_data = json.load(f)
    all_rows.extend(extract_efp_annual_cycle(efp_data, model_name))

df = pd.DataFrame(all_rows)
df.to_csv(OUT_FILE, index=False)
print(f'\nSaved {len(df)} rows → {OUT_FILE}')
print(df.head())
