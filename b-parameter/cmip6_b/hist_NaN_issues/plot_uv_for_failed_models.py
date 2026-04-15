import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cartopy.crs as ccrs
import logging

import functions.SIT_functions.SIT_eddy_feedback_functions as eff


# --- Logging setup ---
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# --- Load processing results ---
logging.info('Loading processing results CSV...')
df = pd.read_csv('/home/users/cturrell/documents/eddy_feedback/b-parameter/cmip6_b/hist_NaN_issues/hist_processing_results.csv')

missing_data_models = df.loc[
    df['failure_reason'] == 'all input data is missing', 'model'
].tolist()
logging.info(f'Found {len(missing_data_models)} models with missing data: {missing_data_models}')

base_dir = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
date_ranges = ['1850_2015', '1850_2014', '1950_2015', '1950_2014']
plevs = [250, 500, 850]
variables = ['ucomp', 'vcomp']
save_path = '/home/users/cturrell/documents/eddy_feedback/b-parameter/cmip6_b/hist_NaN_issues/plots/uv_250_500_850'


def find_yearly_data_dir(model):
    for date_range in date_ranges:
        path = os.path.join(base_dir, model, date_range, '6hrPlevPt', 'yearly_data')
        if os.path.isdir(path):
            return path, date_range
    return None, None


# --- Loop over models ---
for i, model in enumerate(missing_data_models, start=1):
    logging.info(f'[{i}/{len(missing_data_models)}] Starting {model}')

    dir_yearly_data, date_range = find_yearly_data_dir(model)

    if dir_yearly_data is None:
        logging.warning(f'  No yearly_data directory found for {model} — skipping.')
        continue

    logging.info(f'  Found data directory: {dir_yearly_data}')

    save_file = os.path.join(save_path, f'{model}_mean_winds.png')
    if os.path.exists(save_file):
        logging.info(f'  Plot already exists, skipping.')
        continue

    start_year = int(date_range.split('_')[0])
    end_year = int(date_range.split('_')[1])

    logging.info(f'  Reading daily averages ({start_year}–{end_year})...')
    ds_day = eff.read_daily_averages(
        yearly_data_dir=dir_yearly_data,
        start_month=start_year,
        end_month=end_year,
        daily_monthly='daily',
        exp_type='cmip6'
    )
    logging.info(f'  Dataset loaded: {dict(ds_day.sizes)}')

    # --- Plot ---
    logging.info(f'  Generating plot...')
    fig, axes = plt.subplots(
        nrows=len(plevs), ncols=len(variables) + 1,
        figsize=(18, 12),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    for row, plev in enumerate(plevs):
        logging.info(f'    Plotting {plev} hPa...')

        # NaN frequency — same mask for both variables, compute once per pressure level
        nan_freq = ds_day['ucomp'].sel(pfull=plev, method='nearest').isnull().mean(dim='time')
        logging.info(f'      NaN frequency range: {float(nan_freq.min()):.3f} – {float(nan_freq.max()):.3f}')

        for col, var in enumerate(variables):
            logging.info(f'Processing: {var}')
            ax = axes[row, col]

            logging.info(f'Calculing all time mean for var: {var}...')
            da = ds_day[var].sel(pfull=plev, method='nearest').mean(dim='time')
            logging.info(f'Mean Calculated for {var}. Now plotting...')

            im = ax.pcolormesh(
                da.lon, da.lat, da.values,
                cmap='RdBu_r', shading='auto',
                transform=ccrs.PlateCarree()
            )
            # Stipple where any NaNs occur
            ax.contourf(
                nan_freq.lon, nan_freq.lat, nan_freq.values,
                levels=[0.0, 1.0], hatches=[None, '...'],
                colors='none', transform=ccrs.PlateCarree()
            )
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label=f'{var} [m/s]')
            ax.coastlines(linewidth=0.8)
            ax.set_title(f'{var} | {plev} hPa')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            logging.info(f'Plotting {var} completed.')

        # NaN frequency column
        ax_nan = axes[row, 2]

        im_nan = ax_nan.pcolormesh(
            nan_freq.lon, nan_freq.lat, nan_freq.values,
            cmap='Reds', shading='auto', vmin=0, vmax=1,
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(im_nan, ax=ax_nan, orientation='horizontal', pad=0.05, label='NaN frequency')
        ax_nan.coastlines(linewidth=0.8)
        ax_nan.set_title(f'NaN frequency | {plev} hPa')
        ax_nan.set_xlabel('Longitude')
        ax_nan.set_ylabel('Latitude')

    fig.suptitle(f'{model} — All-time mean winds ({date_range})', fontsize=14, y=1.02)
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  Saved plot to {save_file}')

logging.info('All models complete.')
