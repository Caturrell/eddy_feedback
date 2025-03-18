import xarray as xr 
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

import functions.eddy_feedback as ef 
import functions.data_wrangling as data 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

path_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/efp_pd_non-regridded'

# extract model names
files = os.listdir(path_dir)
models = [os.path.basename(f).split('_')[0] for f in files]
models.sort()

models.remove('CESM1-WACCM-SC')
models.remove('E3SMv1')

# Create dictionary containing each model name and dataset
pamip_datasets = {}
for model in models:
    logging.info(f'Loading data for model: {model}')
    # create file path by joining directory and model name
    file_path = os.path.join(path_dir, f'{model}_*.nc')
    # open xarray dataset
    ds = xr.open_mfdataset(file_path, parallel=True, chunks={'time':31})
    # Add dataset to dictionary with required model name
    pamip_datasets[model] = ds 

save_dir = '/home/links/ct715/eddy_feedback/chapter1/saffin/plots'

def process_and_plot(model, season, lat_min, lat_max, hemisphere):
    logging.info(f'Processing data for model: {model} ({hemisphere})')
    # manipulate data
    ds = data.seasonal_mean(pamip_datasets[model], season=season)
    ds = data.data_checker1000(ds)
    ds = ds.mean('time')
        
    # define zonal mean zonal wind
    u = ds.ubar
    u = u.sel(level=500.)
    u = u.where(u.lat >= lat_min, drop=True)
    u = u.where(u.lat <= lat_max, drop=True)
    # flip axes so time on x-axis
    u = u.transpose()

    # set divFy
    div1 = ds.divFy
    div1 = div1.sel(level=500.)
    div1 = div1.where(div1.lat >= lat_min, drop=True)
    div1 = div1.where(div1.lat <= lat_max, drop=True)
    # flip axes so time on x-axis
    div1 = div1.transpose()

    # covariance
    uanom = (u - u.mean('ens_ax'))
    div1anom = (div1 - div1.mean('ens_ax'))
    prod = uanom * div1anom

    # correlation
    ustd = u.std()
    div1std = div1.std()
    corr = prod / (ustd * div1std)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(12,6))

    # plot ubar
    u.plot(ax=axs[0,0], cmap='magma_r', vmin=0, vmax=25,
        cbar_kwargs={'label': 'ms$^{-1}$', 'ticks':[0, 5, 10, 15, 20]})
    axs[0,0].set_yticks(np.arange(lat_min, lat_max+10, 10))
    axs[0,0].set_yticklabels([f'{int(lat)}N' if lat > 0 else f'{int(-lat)}S' for lat in np.arange(lat_min, lat_max+10, 10)])
    axs[0,0].set_title('(a) $\\overline{u}$', fontsize=14)
    axs[0,0].set_ylabel('Latitude', fontsize=12)
    axs[0,0].set_xlabel('')

    # plot div1
    div1.plot(ax=axs[0,1], cmap='PuOr_r',
        cbar_kwargs={'label': 'ms$^{-2}$'})
    axs[0,1].set_title('(b) $\\nabla_\\phi F_\\phi$', fontsize=14)
    axs[0,1].set_ylabel('')
    axs[0,1].set_xlabel('')

    # plot covariance
    prod.plot(ax=axs[1,0], cmap='BrBG_r', cbar_kwargs={'label': 'm$^2$ s$^{-3}$'})
    axs[1,0].set_title('(c) Covariance', fontsize=14)
    axs[1,0].set_ylabel('Latitude', fontsize=12)
    axs[1,0].set_xlabel('Year', fontsize=12)

    # plot correlation
    corr.plot(ax=axs[1,1], cmap='BrBG_r', vmin=-1., vmax=1.,
            cbar_kwargs={'ticks':[-1,-0.5,0,0.5,1]})
    axs[1,1].set_title('(d) Correlation', fontsize=14)
    axs[1,1].set_ylabel('')
    axs[1,1].set_xlabel('Year', fontsize=12)

    plt.tight_layout()
    fig.suptitle(f'{model} ({hemisphere})', fontsize=16)
    plot_path = f'{save_dir}/{hemisphere}/{model}_efp_saffin.pdf'
    plt.savefig(plot_path)
    logging.info(f'Saved plot for model {model} ({hemisphere}) at {plot_path}')
    plt.close()

for model in models:
    process_and_plot(model, 'djf', 25, 75, 'nh')
    process_and_plot(model, 'jas', -75, -25, 'sh')