import xarray as xr
import numpy as np
import os

# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR = '/home/links/ct715/data_storage/reanalysis/era5/6h_era5'
data_vars = ['u250', 'v250', 'u500', 'v500', 'u850', 'v850']

# ── main loop ─────────────────────────────────────────────────────────────────

for var in data_vars:

    print(f"\n{'='*50}")
    print(f"Processing: {var}")

    # source directory: e.g. 6h_era5/u250/
    src_dir = os.path.join(DATA_DIR, var)

    # save directory: e.g. 6h_era5/u/u250/
    uv_letter = var[0]  # 'u' or 'v'
    save_dir = os.path.join(DATA_DIR, uv_letter, var)
    os.makedirs(save_dir, exist_ok=True)

    # open all yearly files as one dataset
    ds = xr.open_mfdataset(os.path.join(src_dir, '*.nc'), combine='by_coords')
    print(f"  Loaded: {ds.dims['time']} timesteps")

    # ── duplicate check ──────────────────────────────────────────────────────
    times = ds['time'].values
    n_dupes = len(times) - len(set(times))
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes} duplicate timestep(s) found — dropping duplicates")
        _, unique_idx = np.unique(times, return_index=True)
        ds = ds.isel(time=unique_idx)
    else:
        print(f"  No duplicate timesteps found")

    # ── split by year and save ───────────────────────────────────────────────
    years = sorted(set(ds['time'].dt.year.values))
    print(f"  Years: {years[0]}–{years[-1]} ({len(years)} total)")

    for year in years:
        ds_year = ds.sel(time=ds['time'].dt.year == year)
        out_path = os.path.join(save_dir, f'ERA5_{var}_{year}.nc')
        ds_year.to_netcdf(out_path)
        print(f"    Saved: {out_path}  ({ds_year.dims['time']} timesteps)")

    ds.close()
    print(f"  Done: {var}")
