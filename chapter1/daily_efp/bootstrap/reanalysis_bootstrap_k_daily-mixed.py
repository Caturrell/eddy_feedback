#!/usr/bin/env python3
"""
Bootstrap EFP for multiple reanalysis configurations and plot boxplots.
Saves results as: {period}_{time_freq}_{var}_{season}-jra55_efp.npy
"""

import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import functions.eddy_feedback as ef
import functions.data_wrangling as dw

# ---------- SETTINGS ----------
NUM_SAMPLES = 1000  # bootstrap samples per case
RANDOM_SEED = 42
# --------------------------------

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging is set up.')


def bootstrap_indices(n, sample_size=None, num_samples=1000):
    """Return array of indices (num_samples x sample_size) for bootstrapping along time dim."""
    sample_size = sample_size or n
    idx = np.random.choice(n, size=(num_samples, sample_size), replace=True)
    return idx


def bootstrap_resampling_ds(ds, sample_size=None, num_samples=1000):
    """
    Produce list of ds subsets resampled along time using bootstrap indices.
    Returns a list of xarray objects each with 'time' dimension length == sample_size.
    """
    logging.debug('Starting bootstrap resampling of dataset with time-size %d', ds.sizes.get('time', 0))
    n = ds.sizes['time']
    indices = bootstrap_indices(n, sample_size=sample_size, num_samples=num_samples)
    # Return list of datasets selected by each row of indices
    return [ds.isel(time=inds) for inds in indices]


def process_reanalysis_bootstrap(ds, which_div1, time_freq, season, sample_size=None,
                                 output_base='.', is_southern=False, num_samples=1000):
    """
    Bootstraps EFP for an xarray dataset `ds` (already sliced to desired time window).
    Returns a 1D numpy array of length num_samples containing EFP values.
    """
    logging.info("Bootstrapping EFP for %s | freq=%s | season=%s | southern=%s | num_samples=%d",
                 which_div1, time_freq, season, is_southern, num_samples)

    # produce bootstrap resampled datasets
    bootstrap_sets = bootstrap_resampling_ds(ds, sample_size=sample_size, num_samples=num_samples)

    efp_values = []
    for i, sample_ds in enumerate(bootstrap_sets):
        if (i + 1) % 100 == 0:
            logging.info("Processing bootstrap sample %d/%d (%s %s)", i+1, num_samples, time_freq, season)
        # calculate EFP for this resample (assumes ef.calculate_efp returns a scalar)
        try:
            val = ef.calculate_efp(sample_ds, which_div1=which_div1, data_type='reanalysis',
                                   calc_south_hemis=is_southern, bootstrapping=True)
        except Exception as e:
            logging.exception("Error calculating EFP on bootstrap sample %d: %s", i, e)
            # append NaN to preserve length
            val = np.nan
        efp_values.append(val)

    efp_values = np.asarray(efp_values, dtype=float)

    # Save results
    var_name = which_div1
    period_part = output_base.split(os.sep)[-2] if output_base else ''
    save_fname = f"{period_part}_{time_freq}_{var_name}_{season}-jra55_efp.npy"
    save_path = os.path.join(output_base, save_fname)
    os.makedirs(output_base, exist_ok=True)
    np.save(save_path, efp_values)
    logging.info("Saved %d EFP bootstraps to %s", efp_values.size, save_path)

    return efp_values


def load_and_concat_yearly_files(path_dir, time_slice=None, decode_times=True, chunks={'time': 31}):
    """
    Load all yearly .nc files in path_dir and concatenate them along time.
    Then apply time_slice if given (a slice object or tuple of start,end strings).
    Returns an xarray Dataset or raises if none found.
    """
    logging.info("Loading files from %s", path_dir)
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"Directory not found: {path_dir}")

    files = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.endswith('.nc')])
    if len(files) == 0:
        raise FileNotFoundError(f"No .nc files found in {path_dir}")

    logging.info("Found %d .nc files; example: %s", len(files), os.path.basename(files[0]))

    # open_mfdataset; use combine='by_coords' for yearly files that align on time coords
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True, decode_times=decode_times, chunks=chunks)

    # apply time_slice if provided (slice('1958-01-01', '2016-12-31') or tuple)
    if time_slice is not None:
        logging.info("Applying time slice %s", repr(time_slice))
        ds = ds.sel(time=time_slice)

    return ds


def make_boxplots(save_dir, djf_results, jas_results, case_labels, out_figpath=None):
    """
    djf_results and jas_results: list of arrays (each array = bootstrap values for one case)
    case_labels: list of strings matching the results order
    Produces a two-panel boxplot figure (upper DJF, lower JAS).
    """
    logging.info("Making boxplots for %d cases per hemisphere", len(case_labels))
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

    # DJF
    axes[0].boxplot(djf_results, labels=case_labels, showfliers=False)
    axes[0].set_title('DJF EFP Bootstrap Distributions (Northern Hemisphere)')
    axes[0].set_ylabel('EFP')

    # JAS
    axes[1].boxplot(jas_results, labels=case_labels, showfliers=False)
    axes[1].set_title('JAS EFP Bootstrap Distributions (Southern Hemisphere)')
    axes[1].set_ylabel('EFP')
    axes[1].set_xticklabels(case_labels, rotation=45, ha='right')

    plt.tight_layout()
    if out_figpath:
        os.makedirs(os.path.dirname(out_figpath), exist_ok=True)
        fig.savefig(out_figpath, dpi=150)
        logging.info("Saved boxplot figure to %s", out_figpath)
    else:
        plt.show()
    plt.close(fig)


def main():
    np.random.seed(RANDOM_SEED)
    setup_logging()

    # user-specified paths (adapt if needed)
    data_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily'
    save_dir = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/data/reanalysis'

    data_folders = {
        '6h': 'k123_6h_QG_epfluxes',
        'daily': 'k123_daily_QG_epfluxes'
    }

    variables = ['div1_QG_123', 'div1_QG_gt3', 'div1_QG']

    time_slice_map = {
        '1958_2016': slice('1958-01-01', '2016-12-31'),
        '1979_2016': slice('1979-01-01', '2016-12-31')
    }

    # Collect results for plotting
    case_labels = []          # order: len = 12 (2 freq * 2 slices * 3 vars)
    djf_collections = []      # list of arrays (bootstrap samples) for DJF
    jas_collections = []      # list of arrays for JAS

    # Loop over combinations -> this produces 12 cases
    for time_freq, folder in data_folders.items():
        for var in variables:
            for period_key, time_sel in time_slice_map.items():
                logging.info("Processing combination: freq=%s | var=%s | period=%s", time_freq, var, period_key)

                path_dir = os.path.join(data_dir, folder)
                # Each variable expected to be a subpath under folder (based on your first script)
                var_dir = os.path.join(path_dir, var)
                if not os.path.isdir(var_dir):
                    # If variables are not separate subfolders, try files directly under folder
                    var_dir = path_dir

                # build output base dir per specification (we'll use this as folder to save .npy)
                output_base = os.path.join(save_dir, f"{period_key}")
                os.makedirs(output_base, exist_ok=True)

                # Load and concat yearly files, then select time slice
                try:
                    ds = load_and_concat_yearly_files(var_dir, time_slice=time_sel)
                except Exception as e:
                    logging.exception("Failed to load dataset for %s: %s", var_dir, e)
                    # append empty arrays to keep ordering but skip
                    case_name = f"{period_key}_{time_freq}_{var}"
                    case_labels.append(case_name)
                    djf_collections.append(np.array([]))
                    jas_collections.append(np.array([]))
                    continue

                # compute seasonal means using your dw.seasonal_mean function
                logging.info("Computing seasonal means for DJF and JAS")
                ds_djf = dw.seasonal_mean(ds, season='djf')
                ds_jas = dw.seasonal_mean(ds, season='jas')

                # Now bootstrap for this case
                case_name = f"{period_key}_{time_freq}_{var}"
                case_labels.append(case_name)

                djf_boot = process_reanalysis_bootstrap(
                    ds_djf, which_div1=var, time_freq=time_freq, season='djf',
                    sample_size=len(time_sel), output_base=output_base, is_southern=False, num_samples=NUM_SAMPLES
                )
                jas_boot = process_reanalysis_bootstrap(
                    ds_jas, which_div1=var, time_freq=time_freq, season='jas',
                    sample_size=len(time_sel), output_base=output_base, is_southern=True, num_samples=NUM_SAMPLES
                )

                djf_collections.append(djf_boot)
                jas_collections.append(jas_boot)

    # Now create a boxplot figure with 12 boxes per hemisphere
    fig_path = os.path.join(save_dir, 'figures', 'jra55_efp_bootstrap_boxplots.png')
    make_boxplots(save_dir, djf_collections, jas_collections, case_labels, out_figpath=fig_path)

    # Save a summary dataframe (optional) for downstream use
    flat_rows = []
    for label, arr in zip(case_labels, djf_collections):
        flat_rows.append({'case': label, 'season': 'DJF', 'efp_mean': np.nanmean(arr), 'n_boot': arr.size})
    for label, arr in zip(case_labels, jas_collections):
        flat_rows.append({'case': label, 'season': 'JAS', 'efp_mean': np.nanmean(arr), 'n_boot': arr.size})

    summary_df = pd.DataFrame(flat_rows)
    summary_csv = os.path.join(save_dir, 'jra55_efp_bootstrap_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    logging.info("Saved summary CSV to %s", summary_csv)


if __name__ == "__main__":
    main()
