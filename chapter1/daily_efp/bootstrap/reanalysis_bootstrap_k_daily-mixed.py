#!/usr/bin/env python3
"""
Bootstrap EFP for multiple reanalysis configurations and plot boxplots.

Processes both full-column and 500 hPa EFP calculations in a single run.

File structure: {efp_type}/{time_period}/{time_freq}/{var}_{season}-jra55_efp.npy
  - efp_type: 'efp_full' (full column) or 'efp_500' (500 hPa only)
  - time_period: '1958_2016' or '1979_2016'
  - time_freq: '6h' or 'daily'
  - var: divergence variable (div1_QG_123, div1_QG_gt3, div1_QG)
  - season: 'djf' or 'jas'

Total outputs: 2 EFP types × 2 time periods × 2 frequencies × 3 variables × 2 seasons = 48 files
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
EFP_TYPES = ['efp_full', 'efp_500']  # List of EFP types to calculate
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


def process_reanalysis_bootstrap(ds, which_div1, time_freq, period_key, season,
                                 save_dir, efp_type='efp_full', sample_size=None,
                                 is_southern=False, num_samples=1000, slice_500hPa=False):
    """
    Bootstraps EFP for an xarray dataset `ds` (already sliced to desired time window and pressure level).
    Returns a 1D numpy array of length num_samples containing EFP values.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset already sliced to desired time range (and pressure level if applicable)
    which_div1 : str
        Divergence variable name
    time_freq : str
        '6h' or 'daily'
    period_key : str
        Time period identifier (e.g., '1958_2016')
    season : str
        'djf' or 'jas'
    save_dir : str
        Base directory for saving results
    efp_type : str
        'efp_full' (full column) or 'efp_500' (500 hPa only)
    sample_size : int, optional
        Bootstrap sample size (default: length of time dimension)
    is_southern : bool
        Whether calculating for Southern Hemisphere
    num_samples : int
        Number of bootstrap samples
    
    Returns
    -------
    efp_values : numpy.ndarray
        Array of bootstrap EFP values
    """
    logging.info("Bootstrapping EFP for %s | freq=%s | period=%s | season=%s | southern=%s | type=%s | num_samples=%d",
                 which_div1, time_freq, period_key, season, is_southern, efp_type, num_samples)

    # produce bootstrap resampled datasets
    bootstrap_sets = bootstrap_resampling_ds(ds, sample_size=sample_size, num_samples=num_samples)

    efp_values = []
    for i, sample_ds in enumerate(bootstrap_sets):
        if (i + 1) % 100 == 0:
            logging.info("Processing bootstrap sample %d/%d (%s %s %s)", 
                        i+1, num_samples, time_freq, period_key, season)
        # calculate EFP for this resample (data already sliced at pressure level if needed)
        try:
            val = ef.calculate_efp(sample_ds, which_div1=which_div1, data_type='reanalysis',
                                  calc_south_hemis=is_southern, bootstrapping=True,
                                  slice_500hPa=slice_500hPa)
        except Exception as e:
            logging.exception("Error calculating EFP on bootstrap sample %d: %s", i, e)
            # append NaN to preserve length
            val = np.nan
        efp_values.append(val)

    efp_values = np.asarray(efp_values, dtype=float)

    # Build hierarchical save path: {efp_type}/{time_period}/{time_freq}/
    output_dir = os.path.join(save_dir, efp_type, period_key, time_freq)
    os.makedirs(output_dir, exist_ok=True)
    
    # Simplified filename: just var and season (path contains other info)
    save_fname = f"{which_div1}_{season}-jra55_efp.npy"
    save_path = os.path.join(output_dir, save_fname)
    
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


def make_boxplots(save_dir, efp_type, djf_results, jas_results, case_labels, out_figpath=None):
    """
    djf_results and jas_results: list of arrays (each array = bootstrap values for one case)
    case_labels: list of strings matching the results order
    Produces a two-panel boxplot figure (upper DJF, lower JAS).
    """
    logging.info("Making boxplots for %d cases per hemisphere (type=%s)", len(case_labels), efp_type)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

    period_colours = {'1958': 'steelblue', '1979': 'darkorange'}
    box_colours = ['steelblue' if '1958' in lbl else 'darkorange' for lbl in case_labels]

    def colour_boxes(bp, colours):
        for patch, colour in zip(bp['boxes'], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

    # DJF
    bp0 = axes[0].boxplot(djf_results, tick_labels=case_labels, showfliers=False, patch_artist=True)
    colour_boxes(bp0, box_colours)
    axes[0].set_title(f'DJF EFP Bootstrap Distributions (Northern Hemisphere) - {efp_type}')
    axes[0].set_ylabel('EFP')
    axes[0].grid(True, alpha=0.3)

    # JAS
    bp1 = axes[1].boxplot(jas_results, tick_labels=case_labels, showfliers=False, patch_artist=True)
    colour_boxes(bp1, box_colours)
    axes[1].set_title(f'JAS EFP Bootstrap Distributions (Southern Hemisphere) - {efp_type}')
    axes[1].set_ylabel('EFP')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)

    # Separator between 6h and daily sections (6 boxes each)
    n_6h = sum(1 for lbl in case_labels if '6h' in lbl)
    sep_x = n_6h + 0.5
    for ax in axes:
        ax.axvline(x=sep_x, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ymin, ymax = ax.get_ylim()
        text_y = ymax - 0.02 * (ymax - ymin)
        ax.text(sep_x / 2, text_y, '6-hourly', ha='center', va='top', fontsize=10, style='italic')
        ax.text((sep_x + len(case_labels) + 1) / 2, text_y, 'Daily', ha='center', va='top', fontsize=10, style='italic')

    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.7, label=f'{p}_2016')
                      for p, c in period_colours.items()]
    axes[0].legend(handles=legend_handles)

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

    # Loop over EFP types (efp_full and efp_500)
    for efp_type in EFP_TYPES:
        logging.info("=" * 80)
        logging.info("STARTING EFP TYPE: %s", efp_type)
        logging.info("=" * 80)
        
        slice_500hPa = efp_type == 'efp_500'

        # Collect results for plotting
        case_labels = []          # order: len = 12 (2 freq * 2 slices * 3 vars)
        djf_collections = []      # list of arrays (bootstrap samples) for DJF
        jas_collections = []      # list of arrays for JAS

        # Loop over combinations -> this produces 12 cases per EFP type
        for time_freq, folder in data_folders.items():
            for var in variables:
                for period_key, time_sel in time_slice_map.items():
                    logging.info("Processing combination: freq=%s | var=%s | period=%s | efp_type=%s",
                               time_freq, var, period_key, efp_type)

                    path_dir = os.path.join(data_dir, folder)
                    # Each variable expected to be a subpath under folder (based on your first script)
                    var_dir = os.path.join(path_dir, var)
                    if not os.path.isdir(var_dir):
                        # If variables are not separate subfolders, try files directly under folder
                        var_dir = path_dir

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

                    djf_path = os.path.join(save_dir, efp_type, period_key, time_freq, f"{var}_djf-jra55_efp.npy")
                    if os.path.exists(djf_path):
                        logging.info("Skipping DJF bootstrap — file already exists: %s", djf_path)
                        djf_boot = np.load(djf_path)
                    else:
                        djf_boot = process_reanalysis_bootstrap(
                            ds_djf, which_div1=var, time_freq=time_freq, period_key=period_key,
                            season='djf', save_dir=save_dir, efp_type=efp_type,
                            sample_size=None, is_southern=False, num_samples=NUM_SAMPLES,
                            slice_500hPa=slice_500hPa
                        )

                    jas_path = os.path.join(save_dir, efp_type, period_key, time_freq, f"{var}_jas-jra55_efp.npy")
                    if os.path.exists(jas_path):
                        logging.info("Skipping JAS bootstrap — file already exists: %s", jas_path)
                        jas_boot = np.load(jas_path)
                    else:
                        jas_boot = process_reanalysis_bootstrap(
                            ds_jas, which_div1=var, time_freq=time_freq, period_key=period_key,
                            season='jas', save_dir=save_dir, efp_type=efp_type,
                            sample_size=None, is_southern=True, num_samples=NUM_SAMPLES,
                            slice_500hPa=slice_500hPa
                        )

                    djf_collections.append(djf_boot)
                    jas_collections.append(jas_boot)

        # Now create a boxplot figure with 12 boxes per hemisphere for this EFP type
        # Save figures in efp_type subdirectory
        fig_dir = os.path.join(save_dir, efp_type, 'figures')
        fig_path = os.path.join(fig_dir, f'jra55_efp_bootstrap_boxplots_{efp_type}.png')
        make_boxplots(save_dir, efp_type, djf_collections, jas_collections, case_labels, out_figpath=fig_path)

        # Save a summary dataframe (optional) for downstream use
        flat_rows = []
        for label, arr in zip(case_labels, djf_collections):
            flat_rows.append({
                'case': label,
                'season': 'DJF',
                'efp_type': efp_type,
                'efp_mean': np.nanmean(arr),
                'efp_std': np.nanstd(arr),
                'efp_median': np.nanmedian(arr),
                'n_boot': arr.size
            })
        for label, arr in zip(case_labels, jas_collections):
            flat_rows.append({
                'case': label,
                'season': 'JAS',
                'efp_type': efp_type,
                'efp_mean': np.nanmean(arr),
                'efp_std': np.nanstd(arr),
                'efp_median': np.nanmedian(arr),
                'n_boot': arr.size
            })

        summary_df = pd.DataFrame(flat_rows)
        summary_csv = os.path.join(save_dir, efp_type, f'jra55_efp_bootstrap_summary_{efp_type}.csv')
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        logging.info("Saved summary CSV to %s", summary_csv)
        
        logging.info("=" * 80)
        logging.info("COMPLETED EFP TYPE: %s", efp_type)
        logging.info("=" * 80)
        logging.info("")  # blank line between types


if __name__ == "__main__":
    main()