"""
Collects the b-parameter annual-cycle data behind the sliding_b vs jra55_850
lag-methodology comparison (see b_comparison_sliding_method/compare_b_sliding_method.py):

    jra55_850  - original lag_method (all_plots_true/jra55_850_sit_plots)
    sliding_b  - Simpson-style sliding-segment lag_method='simpson_sliding'
                 (all_plots_true/sliding_b_sit_plots), same underlying
                 1979-2014 non-detrended anomalies, only the lag construction
                 differs (see jra55_250-850_calc_efp_b.py)

Both runs only overlap on the 1979_2014 period, so that's what's used here.

For each level config (level_250_500_850hPa, level_full_100_850) and each
b variant (va = vertically-averaged, native = non-averaged), writes one CSV
covering both methods (a 'method' column distinguishes jra55_850 from
sliding_b), with the three wavenumber bands (div1_QG, div1_QG_123,
div1_QG_gt3) as columns and the 12 overlapping-season annual cycle as rows,
plus an 'all_time' row at the end of each method's block - matching the
format/season ordering of z_fig5e_b_annual-cycle.py (JJA-start, wrapping to
MJJ). CSVs are written to data/sliding_vs_fixed_window/.

Source data: <method>_sit_plots/1979_2014/6hourly/<level_dir>/b_dataset.nc
(see b_fit_simpson_2013 in functions/SIT_functions/SIT_eddy_feedback_functions.py).
Each 'ucomp{va_str}_{var}{va_str}_b_s_{time_frame}' entry is a 15-length lag
array with lags 0-6 NaN by construction; b is np.nanmean over that array
(lags 7-14).

Southern hemisphere only.
"""

import os
import numpy as np
import pandas as pd
import xarray as xar

script_dir = os.path.dirname(os.path.abspath(__file__))
all_plots_true_dir = os.path.join(script_dir, '..')

hemisphere = 's'

methods = {
    'jra55_850': 'jra55_850_sit_plots',
    'sliding_b': 'sliding_b_sit_plots',
}

level_dirs = {
    '250_500_850hPa': 'level_250_500_850hPa',
    'full_100_850': 'level_full_100_850',
}

variants = {
    'va': '_va',
    'native': '',
}

vars_to_analyse = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']

# Annual-cycle season order (JJA-start) with the matching centre-month letter,
# as used in z_fig5e_b_annual-cycle.py.
annual_cycle_seasons = ['JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ',
                         'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ']
centre_month_labels = ['J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J']


def get_b(b_ds, va_str, var_to_analyse, time_frame):
    name = f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'
    return float(np.nanmean(b_ds[name].values))


data_dir = os.path.join(script_dir, 'data', 'sliding_vs_fixed_window')
os.makedirs(data_dir, exist_ok=True)

b_datasets = {}
for level_tag, level_dir in level_dirs.items():
    for method_tag, method_dir in methods.items():
        b_dataset_path = os.path.join(
            all_plots_true_dir, method_dir, '1979_2014', '6hourly', level_dir, 'b_dataset.nc'
        )
        b_datasets[(level_tag, method_tag)] = xar.open_dataset(b_dataset_path)

for level_tag in level_dirs:
    for variant_tag, va_str in variants.items():
        rows = []
        for method_tag in methods:
            b_ds = b_datasets[(level_tag, method_tag)]

            for season, centre_month in zip(annual_cycle_seasons, centre_month_labels):
                row = {'method': method_tag, 'time_frame': season, 'centre_month': centre_month}
                for var_to_analyse in vars_to_analyse:
                    row[var_to_analyse] = get_b(b_ds, va_str, var_to_analyse, season)
                rows.append(row)

            all_time_row = {'method': method_tag, 'time_frame': 'all_time', 'centre_month': ''}
            for var_to_analyse in vars_to_analyse:
                all_time_row[var_to_analyse] = get_b(b_ds, va_str, var_to_analyse, 'all_time')
            rows.append(all_time_row)

        df = pd.DataFrame(rows)

        out_file = os.path.join(data_dir, f'b_{level_tag}_{variant_tag}_s_annual_cycle.csv')
        df.to_csv(out_file, index=False)
        print(f'Saved {out_file}')

for b_ds in b_datasets.values():
    b_ds.close()
