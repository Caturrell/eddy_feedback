import xarray as xr
import os
import csv

base_data_dir = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
possible_time_spans = ['1850_2015', '1850_2014', '1950_2015', '1950_2014']
level_type = '6hrPlevPt'
start_month = 1979
end_month = 2015

# model_list = [
#     'AWI-ESM-1-1-LR',
#     'CNRM-CM6-1-HR',
#     'EC-Earth3-Veg-LR',
#     'FGOALS-f3-L',
#     'GISS-E2-1-G',
#     'HadGEM3-GC31-MM',
#     'IPSL-CM6A-LR',
#     'IPSL-CM6A-LR-INCA',
#     'KACE-1-0-G',
#     'MIROC6',
#     'MRI-ESM2-0',
#     'SAM0-UNICON',
# ]

model_list = sorted(os.listdir(base_data_dir))

eof_vars = ['ucomp', 'div1_QG', 'div1_QG_123', 'div1_QG_gt3']

output_csv = '/home/users/cturrell/documents/eddy_feedback/b-parameter/cmip6_b/hist_NaN_issues/nan_counts_all_models.csv'

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'var', 'perc_nan'])

    for model_name in model_list:
        time_span = None
        for ts in possible_time_spans:
            if os.path.isdir(f'{base_data_dir}/{model_name}/{ts}/{level_type}/yearly_data'):
                time_span = ts
                break
        if time_span is None:
            print(f'{model_name}: no valid time span directory found, skipping')
            continue

        anom_file = f'{base_data_dir}/{model_name}/{time_span}/{level_type}/{start_month}_{end_month}/anoms.nc'
        if not os.path.isfile(anom_file):
            print(f'\n{model_name}: anoms.nc not found, skipping')
            continue

        print(f'\n=== {model_name} ===')
        anom_ds = xr.open_dataset(anom_file)
        for var in eof_vars:
            for suffix in ['_orig', '_anom']:
                key = f'{var}{suffix}'
                if key in anom_ds:
                    n_nans = int(anom_ds[key].isnull().sum().values)
                    total = int(anom_ds[key].size)
                    perc_nan = round(100 * n_nans / total, 4)
                    print(f'  {key}: {n_nans} NaNs / {total} total ({perc_nan:.2f}%)')
                    writer.writerow([model_name, key, perc_nan])
        anom_ds.close()

print(f'\nSaved to {output_csv}')
