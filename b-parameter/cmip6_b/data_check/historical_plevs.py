import xarray as xr
import numpy as np
import os
import glob
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

path = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
output_file = 'historical_plevs.csv'

logging.info(f"Scanning path: {path}")
models = sorted(os.listdir(path))
logging.info(f"Found {len(models)} models: {models}")

results = []

for model in models:
    model_path = os.path.join(path, model)
    yr = os.listdir(model_path)[0]
    logging.info(f"{model}: using directory '{yr}'")

    daily_data_path = os.path.join(model_path, yr, '6hrPlevPt', 'yearly_data')
    files = glob.glob(daily_data_path + '/*_daily_averages.nc')

    if not files:
        logging.warning(f"{model}: no files found, skipping")
        print(f'-   {daily_data_path}')
        print(f'-   {files}')
        continue

    first_file = files[0]
    logging.info(f"{model}: opening {os.path.basename(first_file)}")

    plev_names = ['plev', 'lev', 'pressure', 'pres', 'level', 'pfull']

    ds = xr.open_dataset(first_file)
    plev_name = next((n for n in plev_names if n in ds.coords or n in ds.dims), None)
    if plev_name is None:
        logging.warning(f"{model}: no recognised pressure coordinate (found: {list(ds.coords)}), skipping")
        ds.close()
        continue
    logging.info(f"{model}: pressure coordinate is '{plev_name}'")
    plevs = ds[plev_name].values
    ds.close()

    results.append((model, len(plevs), plevs))
    logging.info(f"{model}: {len(plevs)} pressure levels found")

# Print summary table
col_w = max(len(m) for m, _, _ in results) + 2
header = f"{'Model':<{col_w}} {'N Levels':>9}  Pressure Levels"
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))
for model, n, plevs in results:
    plev_str = ', '.join(f"{p:.0f}" for p in plevs)
    print(f"{model:<{col_w}} {n:>9}  {plev_str}")
print("=" * len(header))

# Save to CSV
logging.info(f"Saving results to {output_file}")
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'n_levels', 'pressure_levels'])
    for model, n, plevs in results:
        writer.writerow([model, n, list(plevs)])

logging.info(f"Done. Results saved to {output_file}")
