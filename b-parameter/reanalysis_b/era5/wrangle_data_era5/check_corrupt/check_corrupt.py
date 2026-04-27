import xarray as xr
import os
import glob
import argparse
import logging
from pathlib import Path

# ── config ───────────────────────────────────────────────────────────────────
DATA_DIR = '/gws/ssde/j25a/arctic_connect/cturrell/reanalysis_data/era5/6h_era5'
OUTPUT_TXT = '/home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5/wrangle_data_era5/check_corrupt/corrupt_files.txt'
# e.g. u250 files are in /gws/ssde/j25a/arctic_connect/cturrell/reanalysis_data/era5/6h_era5/u/u250/
data_vars = ['u250', 'v250', 'u500', 'v500', 'u850', 'v850']    


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
        level=level,
    )


def find_nc_files(data_dir: str, variables: list[str]) -> list[str]:
    """Glob all .nc files for the specified variables."""
    files = []
    for var in variables:
        # variable name without level, e.g. 'u' from 'u500'
        var_letter = ''.join(c for c in var if c.isalpha())
        pattern = os.path.join(data_dir, var_letter, var, '*.nc')
        matched = sorted(glob.glob(pattern))
        logging.info(f'  {var}: {len(matched)} files found  ({pattern})')
        files.extend(matched)
    return files


def is_corrupt(filepath: str) -> tuple[bool, str]:
    """
    Try to open a NetCDF file and load its data into memory.
    Returns (corrupt: bool, reason: str).
    """
    try:
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            # Read first and last timestep of each variable to catch truncation
            # without loading the full file (~6 GB) into memory
            for var in ds.data_vars:
                da = ds[var]
                if 'time' in da.dims:
                    da.isel(time=0).values
                    da.isel(time=-1).values
                else:
                    da.values
        return False, ''
    except Exception as exc:
        return True, str(exc)


def check_files(
    files: list[str],
    output_txt: str,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Iterate over files, attempt to open each one, and record corrupt paths.

    Returns
    -------
    corrupt : list of corrupt file paths
    ok      : list of clean file paths
    """
    corrupt, ok = [], []
    n = len(files)

    for i, fp in enumerate(files, 1):
        logging.info(f'[{i}/{n}]  Checking: {os.path.basename(fp)}')
        bad, reason = is_corrupt(fp)
        if bad:
            logging.warning(f'  ✗  CORRUPT — {reason}')
            corrupt.append(fp)
        else:
            logging.debug(f'  ✓  OK')
            ok.append(fp)

    # ── write results ─────────────────────────────────────────────────────────
    if not dry_run:
        Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt, 'w') as fh:
            fh.write(f'# Corrupt NetCDF files ({len(corrupt)} of {n} checked)\n')
            for fp in corrupt:
                fh.write(fp + '\n')
        logging.info(f'\nResults written to: {output_txt}')
    else:
        logging.info('\n[dry-run] — no output file written.')

    return corrupt, ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check for corrupted NetCDF files in ERA5 data directories.',
    )
    parser.add_argument(
        '--data-dir', default=DATA_DIR,
        help='Root data directory (default: %(default)s)',
    )
    parser.add_argument(
        '--output', default=OUTPUT_TXT,
        help='Path for the corrupt-files list (default: %(default)s)',
    )
    parser.add_argument(
        '--vars', nargs='+', default=data_vars,
        metavar='VAR',
        help='Variable names to check, e.g. u500 v500 (default: %(default)s)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Check files but do not write output',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show OK files as well as corrupt ones',
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    setup_logging(args.verbose)

    logging.info('=== ERA5 NetCDF corruption check ===')
    logging.info(f'Data dir : {args.data_dir}')
    logging.info(f'Variables: {args.vars}')

    files = find_nc_files(args.data_dir, args.vars)
    logging.info(f'Total files to check: {len(files)}\n')

    if not files:
        logging.warning('No files found — check DATA_DIR and variable names.')
    else:
        corrupt, ok = check_files(files, args.output, dry_run=args.dry_run)

        # ── summary ───────────────────────────────────────────────────────────
        print('\n' + '─' * 50)
        print(f'  Checked : {len(files)}')
        print(f'  OK      : {len(ok)}')
        print(f'  Corrupt : {len(corrupt)}')
        if corrupt:
            print('\n  Corrupt files:')
            for fp in corrupt:
                print(f'    {fp}')
        print('─' * 50)