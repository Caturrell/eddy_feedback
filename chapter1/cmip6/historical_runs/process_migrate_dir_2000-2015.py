import logging
import os
import numpy as np

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#==============================================================================================================
# Configuration
#==============================================================================================================

base_dir_output  = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
level_type       = '6hrPlevPt'
old_end_year     = 2000    # nominal end year of the previous run — dirs within old_end_year_tolerance of this will be matched
old_end_year_tolerance = 2  # match dirs ending in e.g. 1998, 1999, 2000, 2001, 2002
new_end_year     = 2015    # end year used in the new run

# Automatically find all model directories in base_dir_output
completed_model_list = sorted([
    d for d in os.listdir(base_dir_output)
    if os.path.isdir(os.path.join(base_dir_output, d))
])
logging.info(f'Found {len(completed_model_list)} model directories: {completed_model_list}')

# These are the subdirectories to migrate within each model's yearly_data folder.
# Add or remove entries if your directory structure differs.
subdirs_to_migrate = [
    'yearly_data',
]

#==============================================================================================================
# Migration
#==============================================================================================================

summary = {
    'migrated':    [],
    'already_exists': [],
    'no_old_dir':  [],
    'errors':      [],
}

for model_name in completed_model_list:

    logging.info(f'{"="*60}')
    logging.info(f'Migrating {model_name}')

    # Identify start year from old directory name by scanning for matching dirs
    model_base = os.path.join(base_dir_output, model_name)
    if not os.path.isdir(model_base):
        logging.warning(f'  Model base directory not found: {model_base}')
        summary['no_old_dir'].append(model_name)
        continue

    # Find old run directory whose end year falls within tolerance of old_end_year
    # e.g. matches 1850_1998, 1850_1999, 1850_2000, 1850_2001, 1850_2002
    old_run_dirs = []
    for d in os.listdir(model_base):
        if not os.path.isdir(os.path.join(model_base, d)):
            continue
        parts = d.split('_')
        if len(parts) != 2:
            continue
        try:
            dir_end_year = int(parts[1])
        except ValueError:
            continue
        if abs(dir_end_year - old_end_year) <= old_end_year_tolerance:
            old_run_dirs.append(d)

    if len(old_run_dirs) == 0:
        logging.warning(f'  No old run directory with end year within {old_end_year_tolerance} of {old_end_year} found for {model_name}')
        summary['no_old_dir'].append(model_name)
        continue

    if len(old_run_dirs) > 1:
        # Pick the one with the largest end year to maximise data carried over
        old_run_dir = sorted(old_run_dirs, key=lambda d: int(d.split('_')[1]), reverse=True)[0]
        logging.warning(f'  Multiple candidate old run directories found for {model_name}: {old_run_dirs}')
        logging.warning(f'  Using the one with the latest end year: {old_run_dir}')
    else:
        old_run_dir = old_run_dirs[0]

    actual_old_end_year = int(old_run_dir.split('_')[1])
    start_year          = old_run_dir.split('_')[0]   # e.g. '1850' from '1850_1999'
    new_run_dir         = f'{start_year}_{new_end_year}'

    if actual_old_end_year != old_end_year:
        logging.info(f'  Note: matched old end year {actual_old_end_year} (expected ~{old_end_year})')

    logging.info(f'  Old run dir : {old_run_dir}  (end year: {actual_old_end_year})')
    logging.info(f'  New run dir : {new_run_dir}')

    model_migrated    = 0
    model_skipped     = 0
    model_errors      = 0

    for subdir in subdirs_to_migrate:

        old_dir = os.path.join(model_base, old_run_dir, level_type, subdir)
        new_dir = os.path.join(model_base, new_run_dir, level_type, subdir)

        if not os.path.isdir(old_dir):
            logging.warning(f'  Old subdirectory not found, skipping: {old_dir}')
            continue

        os.makedirs(new_dir, exist_ok=True)
        logging.info(f'  Migrating: {old_dir}')
        logging.info(f'        --> {new_dir}')

        for fname in sorted(os.listdir(old_dir)):
            src  = os.path.join(old_dir, fname)
            dest = os.path.join(new_dir, fname)

            # If destination is an existing symlink (from a previous migration run),
            # remove it so we can replace it with the real moved file
            if os.path.islink(dest):
                logging.info(f'    Removing old symlink: {fname}')
                os.remove(dest)
            elif os.path.exists(dest):
                logging.debug(f'    Skipping (real file already exists at destination): {fname}')
                model_skipped += 1
                continue

            try:
                os.rename(src, dest)
                logging.info(f'    Moved: {fname}')
                model_migrated += 1
            except Exception as e:
                logging.error(f'    Failed to move {fname}: {e}')
                model_errors += 1

    logging.info(f'  Done: {model_migrated} moved, {model_skipped} already existed, {model_errors} errors')

    # Remove old run directory tree if all subdirs are now empty
    old_run_base = os.path.join(model_base, old_run_dir)
    removed_dirs = []
    kept_dirs    = []
    for dirpath, dirnames, filenames in os.walk(old_run_base, topdown=False):
        if not filenames and not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
                removed_dirs.append(dirpath)
            except Exception as e:
                logging.error(f'  Failed to remove empty dir {dirpath}: {e}')
                kept_dirs.append(dirpath)
        else:
            kept_dirs.append(dirpath)

    if not kept_dirs:
        logging.info(f'  Removed old empty directory tree: {old_run_base}')
        summary.setdefault('removed', []).append(model_name)
    else:
        logging.info(f'  Old directory not fully empty, left in place: {old_run_base}')
        if removed_dirs:
            logging.info(f'  Removed {len(removed_dirs)} empty subdirectories within it')

    if model_errors > 0:
        summary['errors'].append(model_name)
    elif model_migrated > 0:
        summary['migrated'].append(model_name)
    else:
        summary['already_exists'].append(model_name)

#==============================================================================================================
# Summary
#==============================================================================================================

logging.info(f'\n{"="*60}')
logging.info('MIGRATION SUMMARY')
logging.info(f'{"="*60}')
logging.info(f'Successfully moved ({len(summary["migrated"])}):')
for m in summary['migrated']:
    logging.info(f'  {m}')
logging.info(f'Already up to date ({len(summary["already_exists"])}):')
for m in summary['already_exists']:
    logging.info(f'  {m}')
logging.info(f'No old directory found ({len(summary["no_old_dir"])}):')
for m in summary['no_old_dir']:
    logging.info(f'  {m}')
logging.info(f'Errors - manual review needed ({len(summary["errors"])}):')
for m in summary['errors']:
    logging.info(f'  {m}')
logging.info(f'Old directories removed ({len(summary.get("removed", []))}):')
for m in summary.get('removed', []):
    logging.info(f'  {m}')
logging.info(f'{"="*60}')