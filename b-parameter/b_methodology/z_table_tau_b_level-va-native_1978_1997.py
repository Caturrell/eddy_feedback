"""
Computes and tabulates tau and b for the four level-configuration / variant
methods used throughout this directory's z_fig* scripts:
    level configuration: 250,500,850hPa vs full 100-850hPa
    variant:             va (vertically-averaged) vs native (full-field)

Southern Hemisphere, all_time, div1_QG (all wavenumbers) only.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1978_1997/
        6hourly/level_250_500_850hPa/{power_spec.nc, b_dataset.nc}
        6hourly/level_full_100_850/{power_spec.nc, b_dataset.nc}

tau: `phase_diff_tau_fit_3` from power_spec.nc, as used in
    z_fig2_cospec_coher_pdiff_level-va-native.py -- the ratio of the
    low-frequency slope of Im(cospectrum) to the low-frequency intercept of
    Re(cospectrum).

b: mean over lag 7-14 of the b_dataset.nc 'lag'-indexed array (lags 0-6 are
    NaN by construction), as used in fig5_m-regression_b-parameter.py /
    b_fit_simpson_2013 -- ratio of the eddy-forcing-on-wind lag-regression
    slope to the wind's own lag-autoregression slope.
"""

import os
import numpy as np
import pandas as pd
import xarray as xar

script_dir = os.path.dirname(os.path.abspath(__file__))

level_dirs = {
    '250_500_850hPa': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1978_1997',
        '6hourly', 'level_250_500_850hPa'
    ),
    'full_100_850': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1978_1997',
        '6hourly', 'level_full_100_850'
    ),
}

hemisphere = 's'
time_frame = 'all_time'

variant_va_str = {'va': '_va', 'native': ''}
variant_labels = {'va': 'vertically-averaged', 'native': 'non-averaged'}
level_labels = {'250_500_850hPa': '250,500,850hPa', 'full_100_850': '100-850hPa'}


def get_tau(power_spec_ds, va_str):
    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'
    div1_name = f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'
    return float(power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'])


def get_b(b_ds, va_str):
    name = f'ucomp{va_str}_div1_QG{va_str}_b_{hemisphere}_{time_frame}'
    return float(np.nanmean(b_ds[name].values))


# ── Compute ──────────────────────────────────────────────────────────────────

rows = []
for level_key, level_dir in level_dirs.items():
    power_spec_ds = xar.open_dataset(os.path.join(level_dir, 'power_spec.nc'), auto_complex=True)
    b_ds = xar.open_dataset(os.path.join(level_dir, 'b_dataset.nc'))

    for variant_key, va_str in variant_va_str.items():
        rows.append({
            'level_config': level_labels[level_key],
            'variant': variant_labels[variant_key],
            'tau (days)': get_tau(power_spec_ds, va_str),
            'b': get_b(b_ds, va_str),
        })

    power_spec_ds.close()
    b_ds.close()

# ── Print + save ─────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
out_file = os.path.join(data_dir, 'tau_b_level-va-native_jra55_1978_1997.csv')
df.to_csv(out_file, index=False)
print(f'\nSaved table to {out_file}')
