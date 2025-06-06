import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import logging
import pdb

import functions.eddy_feedback as ef      # unchanged
import functions.data_wrangling as data    # unchanged

# ----------‑‑‑ housekeeping ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

path_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/efp_pd_non-regridded'
save_dir = '/home/links/ct715/eddy_feedback/chapter1/saffin/plots'

# model list
models = sorted(
    m.split('_')[0] for m in os.listdir(path_dir)
    if m.endswith('.nc')
)
for bad in ('CESM1-WACCM-SC', 'E3SMv1'):
    if bad in models:
        models.remove(bad)

# ----------‑‑‑ little helpers --------------------------------------------------
def correlation_panel(ds, season, lat_min, lat_max):
    """Return the correlation field that used to go in axs[1,1]."""
    ds = data.data_checker1000(ds)
    
    # calculate EFP
    if season == 'djf':
        print(f'Calculating EFP for {season}')
        efp = ef.calculate_efp(ds, data_type='pamip')
    elif season == 'jas':
        print(f'Calculating EFP for {season}')
        efp = ef.calculate_efp(ds, data_type='pamip', calc_south_hemis=True)
    else:
        raise ValueError('season not recognised')
    
    # subset data
    ds = data.seasonal_mean(ds, season=season)
    ds = ds.mean('time')
    ds = ds.sel(lat=slice(lat_min, lat_max))

    u     = ds.ubar.sel(level=500.)
    div1  = ds.divFy.sel(level=500.)

    u_an      = u - u.mean('ens_ax')
    div1_an   = div1 - div1.mean('ens_ax')
    prod      = u_an * div1_an
    corr      = prod / (u.std() * div1.std())

    return corr, efp


def plot_all_models(season, lat_min, lat_max, hemisphere_tag):
    """Build one figure that shows the correlation heat‑map for every model."""
    n = len(models)
    n_cols = 4                             # 4 wide is a good default
    n_rows = math.ceil(n / n_cols)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        sharey=True,
        figsize=(3 * n_cols, 3 * n_rows)
    )
    axs = axs.flatten()                    # easy 1‑D indexing

    for i, model in enumerate(models):
        logging.info(f'Processing {model} ({hemisphere_tag})')
        ds   = xr.open_mfdataset(f'{path_dir}/{model}_*.nc', parallel=True, chunks={'time': 31})
        corr, efp = correlation_panel(ds, season, lat_min, lat_max)
        corr = corr.transpose()
        efp = np.round(efp, 2)

        corr.plot(
            ax             = axs[i],
            cmap           = 'BrBG_r',
            vmin           = -1.0,
            vmax           = 1.0,
            add_colorbar   = False
        )
        axs[i].set_title(model + f' ({efp})', fontsize=9)
        axs[i].set_xlabel('')              # keep it clean
        axs[i].set_ylabel('' if i else 'Latitude')  # only first gets label
        ds.close()

    # hide any empty axes (if n_models % n_cols != 0)
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axs[j])

    # one shared colour‑bar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm  = plt.cm.ScalarMappable(cmap='BrBG_r', norm=plt.Normalize(-1, 1))
    fig.colorbar(sm, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1],
                 label='Correlation')

    fig.suptitle(
        f'Correlation between $\\overline{{u}}$ and $\\nabla_\\phi F_\\phi$ '
        f'({season.upper()}, {hemisphere_tag.upper()})',
        fontsize=16
    )
    # plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    out_file = f'{save_dir}/{hemisphere_tag}/ALL_MODELS_correlation_{season}.png'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file)
    logging.info(f'Saved combined figure at {out_file}')
    plt.close(fig)

# ----------‑‑‑ run it ----------------------------------------------------------
# northern winter (DJF) & southern winter (JAS) – same lat bounds you used before
plot_all_models('djf',  25,  75, 'nh')
plot_all_models('jas', -75, -25, 'sh')
