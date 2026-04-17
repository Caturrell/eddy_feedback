import xarray as xr
import os
import glob
import argparse

# ── config ───────────────────────────────────────────────────────────────────
DATA_DIR = '/gws/ssde/j25a/arctic_connect/cturrell/reanalysis_data/era5/6h_era5'
OUTPUT_TXT = '/home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5/wrangle_data_era5/check_corrupt/corrupt_files.txt'

# ── parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Check netCDF files for corruption.')
parser.add_argument('--dir', default=DATA_DIR, help='Directory to search for .nc files')
parser.add_argument('--out', default=OUTPUT_TXT, help='Output txt file for corrupt paths')
parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
args = parser.parse_args()

# ── find files ────────────────────────────────────────────────────────────────
pattern = os.path.join(args.dir, '**/*.nc') if args.recursive else os.path.join(args.dir, '*.nc')
nc_files = sorted(glob.glob(pattern, recursive=args.recursive))

print(f"Found {len(nc_files)} .nc files in: {args.dir}")

# ── check each file ───────────────────────────────────────────────────────────
corrupt = []

for i, fpath in enumerate(nc_files, 1):
    try:
        ds = xr.open_dataset(fpath)
        # load a small slice to trigger actual I/O (catches truncated files)
        ds.load()
        ds.close()
        print(f"  [{i}/{len(nc_files)}] OK      {fpath}")
    except Exception as e:
        print(f"  [{i}/{len(nc_files)}] CORRUPT {fpath}  ({e})")
        corrupt.append(fpath)

# ── report ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Checked {len(nc_files)} files — {len(corrupt)} corrupt")

if corrupt:
    with open(args.out, 'w') as f:
        f.write('\n'.join(corrupt) + '\n')
    print(f"Corrupt file paths written to: {args.out}")
else:
    print("No corrupt files found.")
