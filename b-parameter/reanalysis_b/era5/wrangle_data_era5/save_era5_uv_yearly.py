import xarray as xr
import numpy as np
import os
import glob

# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR = '/gws/ssde/j25a/arctic_connect/cturrell/reanalysis_data/era5/6h_era5'
# data_vars = ['u500', 'v500', 'u850', 'v850']
data_vars = ['u250', 'v250']

# ── select variable via SLURM array task ID (or process all if not in array) ──
task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
if task_id is not None:
    data_vars = [data_vars[int(task_id)]]

# ── main loop ─────────────────────────────────────────────────────────────────

for var in data_vars:

    print(f"\n{'='*50}")
    print(f"Processing: {var}")

    # source directory: e.g. 6h_era5/u
    uv_letter = var[0]  # 'u' or 'v'
    src_dir = os.path.join(DATA_DIR, uv_letter)

    # save directory: e.g. 6h_era5/u/u500/
    save_dir = os.path.join(DATA_DIR, uv_letter, var)
    os.makedirs(save_dir, exist_ok=True)

    # process source files one at a time to avoid loading everything into memory
    src_files = sorted(glob.glob(os.path.join(src_dir, f'ERA5_{var}_*.nc')))
    if not src_files:
        print(f"  WARNING: no source files found for {var}, skipping")
        continue
    print(f"  Found {len(src_files)} source file(s)")

    for src_file in src_files:
        print(f"  Opening: {os.path.basename(src_file)}")

        # chunks={'time': 100} streams data rather than loading it all at once
        ds = xr.open_dataset(src_file, chunks={'time': 100})

        # ── duplicate check ──────────────────────────────────────────────────
        times = ds['time'].values
        n_dupes = len(times) - len(set(times))
        if n_dupes > 0:
            print(f"  WARNING: {n_dupes} duplicate timestep(s) found — dropping duplicates")
            _, unique_idx = np.unique(times, return_index=True)
            ds = ds.isel(time=unique_idx)

        # ── split by year and save ───────────────────────────────────────────
        years = sorted(set(ds['time'].dt.year.values))

        for year in years:
            out_path = os.path.join(save_dir, f'ERA5_{var}_{year}.nc')

            if os.path.exists(out_path):
                print(f"    Skipping (exists): ERA5_{var}_{year}.nc")
                continue

            ds_year = ds.sel(time=ds['time'].dt.year == year)
            ds_year.to_netcdf(out_path)
            print(f"    Saved: ERA5_{var}_{year}.nc  ({ds_year.sizes['time']} timesteps)")

        ds.close()

    print(f"  Done: {var}")
