import os
import sys
import logging
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import date, datetime, timedelta

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

MAIN_DIR = '/badc/ecmwf-era5/data/oper/an_ml'
BASE_SAVE_DIR = '/gws/ssde/j25a/arctic_connect/cturrell/reanalysis_data/era5/6h_era5/an_ml_extract'

START_YEAR = 1979
END_YEAR = 2016

VARIABLES = ['u', 'v', 't']
HOURS = ['0000', '0600', '1200', '1800']

TARGET_LEVELS_HPA = np.array([250, 500, 850], dtype=np.float64)

COEFFS_CSV = os.path.join(os.path.dirname(__file__), 'era5_L137_model_levels.csv')

ERA5_1_YEARS = set(range(2000, 2007))


# ── path helpers ──────────────────────────────────────────────────────────────

def candidate_paths(var, d, hhmm):
    """Return the expected CEDA file path for a given variable, date, and time."""
    yyyy, mm, dd = d.strftime('%Y'), d.strftime('%m'), d.strftime('%d')

    if d.year in ERA5_1_YEARS:
        fname = f"ecmwf-era51_oper_an_ml_{d.strftime('%Y%m%d')}{hhmm}.{var}.nc"
        return [os.path.join(MAIN_DIR, yyyy, f"era5.1_{yyyy}_data", mm, dd, fname)]
    else:
        fname = f"ecmwf-era5_oper_an_ml_{d.strftime('%Y%m%d')}{hhmm}.{var}.nc"
        return [os.path.join(MAIN_DIR, yyyy, mm, dd, fname)]


# ── data availability check ───────────────────────────────────────────────────

def check_data_availability(start_year=START_YEAR, end_year=END_YEAR):
    """Check that all expected ERA5 model-level files exist for u, v, t at 00/06/12/18Z."""
    missing = []

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    current = start
    total = 0

    while current <= end:
        for hhmm in HOURS:
            for var in VARIABLES:
                paths = candidate_paths(var, current, hhmm)
                total += 1
                if not any(os.path.isfile(p) for p in paths):
                    missing.append(paths[0])
        current += timedelta(days=1)

    n_missing = len(missing)
    logging.info(f"Checked {total} files ({start_year}–{end_year}, vars={VARIABLES}, times={HOURS})")
    if n_missing == 0:
        logging.info("All files present.")
    else:
        logging.warning(f"{n_missing} file(s) missing.")
        for p in missing:
            logging.warning(f"  MISSING: {p}")

    return missing


# ── hybrid coefficients ───────────────────────────────────────────────────────

def load_hybrid_coefficients(csv_path=COEFFS_CSV):
    """
    Load ERA5 L137 hybrid sigma-pressure coefficients from the ECMWF CSV table.
    Returns a_half (Pa) and b_half (dimensionless) at 138 half-levels (n=0..137).
    CSV columns: n, a[Pa], b, ...
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1, 2))
    a_half = data[:, 0]  # (138,) Pa
    b_half = data[:, 1]  # (138,) dimensionless
    return a_half, b_half


# ── model-level to pressure conversion ───────────────────────────────────────

def compute_full_level_pressure(lnsp_2d, a_half, b_half):
    """
    Compute pressure (Pa) at all 137 full model levels from log surface pressure.
    lnsp_2d: (nlat, nlon)
    Returns: p_full (137, nlat, nlon) in Pa, ordered top-to-bottom.
    """
    ps = np.exp(lnsp_2d)  # (nlat, nlon)
    p_half = a_half[:, None, None] + b_half[:, None, None] * ps[None, :, :]  # (138, nlat, nlon)
    return 0.5 * (p_half[:-1] + p_half[1:])  # (137, nlat, nlon)


def find_bracketing_levels(p_full, target_hpa):
    """
    Find the 0-based model level indices needed to bracket each target pressure
    across every grid point. Returns a sorted list of unique indices.

    p_full:     (137, nlat, nlon) in Pa
    target_hpa: (n_target,) in hPa
    """
    nlev = p_full.shape[0]
    needed = set()

    for p_tgt in target_hpa:
        p_tgt_pa = p_tgt * 100.
        count = np.sum(p_full <= p_tgt_pa, axis=0)  # (nlat, nlon)
        valid = (count >= 1) & (count < nlev)
        if not valid.any():
            logging.warning(f"No valid bracketing levels found for {p_tgt} hPa — skipping.")
            continue
        i0_min = int(np.min(count[valid])) - 1   # lowest index needed
        i1_max = int(np.max(count[valid]))         # highest index needed
        needed.update(range(i0_min, i1_max + 1))

    return sorted(needed)


def interp_log_pressure(data_ml, p_full_pa, target_hpa):
    """
    Vectorized log-pressure interpolation from model levels to target levels.
    Model levels must be ordered top-to-bottom (pressure increasing with index).

    data_ml:    (nlev, nlat, nlon)
    p_full_pa:  (nlev, nlat, nlon) in Pa  — must match data_ml levels
    target_hpa: (n_target,) in hPa
    Returns:    (n_target, nlat, nlon) float32
    """
    nlev, nlat, nlon = data_ml.shape
    n_target = len(target_hpa)
    result = np.full((n_target, nlat, nlon), np.nan, dtype=np.float32)

    log_p_2d = np.log(p_full_pa).reshape(nlev, -1)
    data_2d  = data_ml.astype(np.float64).reshape(nlev, -1)

    for k, p_tgt in enumerate(target_hpa):
        log_t = np.log(p_tgt * 100.)

        count = np.sum(log_p_2d <= log_t, axis=0)  # (ncols,)
        valid = (count >= 1) & (count < nlev)
        if not valid.any():
            continue

        col_idx = np.where(valid)[0]
        i0 = count[col_idx] - 1
        i1 = count[col_idx]

        lp0 = log_p_2d[i0, col_idx]
        lp1 = log_p_2d[i1, col_idx]
        d0  = data_2d[i0, col_idx]
        d1  = data_2d[i1, col_idx]

        denom = lp1 - lp0
        w = np.where(denom != 0, (log_t - lp0) / denom, 0.5)
        result[k].reshape(-1)[col_idx] = (d0 + w * (d1 - d0)).astype(np.float32)

    return result


# ── output file creation ──────────────────────────────────────────────────────

def _create_output_file(out_path, var, lat, lon):
    """Create a new netCDF4 output file with correct dimensions and a single variable."""
    ncfile = nc.Dataset(out_path, 'w', format='NETCDF4')

    ncfile.createDimension('time', None)
    ncfile.createDimension('level', len(TARGET_LEVELS_HPA))
    ncfile.createDimension('latitude', len(lat))
    ncfile.createDimension('longitude', len(lon))

    t_var = ncfile.createVariable('time', 'f8', ('time',))
    t_var.units = 'hours since 1900-01-01 00:00:00'
    t_var.calendar = 'gregorian'

    lev_var = ncfile.createVariable('level', 'f4', ('level',))
    lev_var[:] = TARGET_LEVELS_HPA
    lev_var.units = 'hPa'
    lev_var.long_name = 'pressure level'

    lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
    lat_var[:] = lat
    lat_var.units = 'degrees_north'

    lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
    lon_var[:] = lon
    lon_var.units = 'degrees_east'

    units = {'u': 'm s-1', 'v': 'm s-1', 't': 'K'}
    chunks = (1, len(TARGET_LEVELS_HPA), len(lat), len(lon))
    v = ncfile.createVariable(
        var, 'f4', ('time', 'level', 'latitude', 'longitude'),
        zlib=True, complevel=4, chunksizes=chunks
    )
    v.units = units[var]

    return ncfile


# ── main processing ───────────────────────────────────────────────────────────

def process_year(year, var, a_half, b_half):
    """Load ERA5 model-level data for one variable, interpolate to TARGET_LEVELS_HPA, save yearly file."""
    save_dir = os.path.join(BASE_SAVE_DIR, var)
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"{year}_{var}_6h_snapshots.nc")
    if os.path.exists(out_path):
        logging.info(f"{year} {var}: already exists, skipping.")
        return

    logging.info(f"{year} {var}: starting.")

    first_file = candidate_paths('u', date(year, 1, 1), '0000')[0]
    with xr.open_dataset(first_file) as ds:
        lat = ds['latitude'].values
        lon = ds['longitude'].values

    epoch = datetime(1900, 1, 1)
    ncfile = _create_output_file(out_path, var, lat, lon)
    t_idx = 0

    try:
        current = date(year, 1, 1)
        end     = date(year, 12, 31)

        while current <= end:
            for hhmm in HOURS:
                lnsp_path = candidate_paths('lnsp', current, hhmm)[0]
                var_path  = candidate_paths(var,    current, hhmm)[0]

                with xr.open_dataset(lnsp_path) as lnsp_ds:
                    lnsp_2d = lnsp_ds['lnsp'].squeeze().values  # (nlat, nlon)

                p_full = compute_full_level_pressure(lnsp_2d, a_half, b_half)  # (137, nlat, nlon)

                idx = find_bracketing_levels(p_full, TARGET_LEVELS_HPA)
                era5_levels = [i + 1 for i in idx]

                with xr.open_dataset(var_path) as ds:
                    var_sub = ds[var].sel(level=era5_levels).squeeze().values  # (n_sub, nlat, nlon)

                p_sub = p_full[idx]

                ncfile.variables[var][t_idx] = interp_log_pressure(var_sub, p_sub, TARGET_LEVELS_HPA)

                hh = int(hhmm[:2])
                hours = (datetime(current.year, current.month, current.day) - epoch).total_seconds() / 3600 + hh
                ncfile.variables['time'][t_idx] = hours

                t_idx += 1

            current += timedelta(days=1)

    except Exception:
        ncfile.close()
        os.remove(out_path)
        raise

    ncfile.close()
    logging.info(f"{year} {var}: saved {t_idx} timesteps → {out_path}")


if __name__ == '__main__':
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', sys.argv[1] if len(sys.argv) > 1 else 0))
    var = VARIABLES[task_id]

    logging.info(f"Processing variable: {var} (task {task_id})")

    a_half, b_half = load_hybrid_coefficients()

    for year in range(START_YEAR, END_YEAR + 1):
        process_year(year, var, a_half, b_half)
