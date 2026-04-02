import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import functions.eddy_feedback as ef
import functions.data_wrangling as dw

path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/processed_efp'
data_6h = os.path.join(path, 'k123_6h_ubar_epf-pr-QG_1MS_1958-2016.nc')

ds_6h = xr.open_dataset(data_6h)
ds_6h = ds_6h[['ubar', 'div1_QG', 'div1_QG_123', 'div1_QG_gt3']]

djf = dw.seasonal_mean(ds_6h, season='djf')
jas = dw.seasonal_mean(ds_6h, season='jas')

div1_vars = {
    'div1_QG':     'Full QG',
    'div1_QG_123': 'QG k=1,2,3',
    'div1_QG_gt3': 'QG k>3',
}

rows = {
    'DJF': {'ds': djf, 'lat_min': 25.,  'lat_max': 75.,
            'ticks': np.arange(30, 80, 10),
            'ticklabels': ['30N', '40N', '50N', '60N', '70N']},
    'JAS': {'ds': jas, 'lat_min': -75., 'lat_max': -25.,
            'ticks': np.arange(-70, -20, 10),
            'ticklabels': ['70S', '60S', '50S', '40S', '30S']},
}

fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey='row')

for row_idx, (season_name, p) in enumerate(rows.items()):
    ds = p['ds']
    u = ds.ubar.sel(level=500.)
    u = u.where(ds.lat >= p['lat_min'], drop=True).where(ds.lat <= p['lat_max'], drop=True)
    uanom = u - u.mean('time')

    for col_idx, (var_name, var_label) in enumerate(div1_vars.items()):
        ax = axes[row_idx, col_idx]

        div1 = ds[var_name].sel(level=500.)
        div1 = div1.where(ds.lat >= p['lat_min'], drop=True).where(ds.lat <= p['lat_max'], drop=True)
        div1anom = div1 - div1.mean('time')

        prod = uanom * div1anom
        corr = (prod / (u.std() * div1.std())).transpose()

        im = corr.plot(ax=ax, cmap='BrBG_r', vmin=-1., vmax=1., add_colorbar=False)

        ax.set_yticks(p['ticks'])
        ax.set_yticklabels(p['ticklabels'] if col_idx == 0 else [])
        ax.set_ylabel('Latitude' if col_idx == 0 else '')
        ax.set_xlabel('Year')

        # Column headers on top row only, season labels on left column only
        if row_idx == 0:
            ax.set_title(var_label)
        else:
            ax.set_title('')

        if col_idx == 0:
            ax.set_ylabel(f'{season_name}\nLatitude')

# Shared colorbar on the right
fig.colorbar(im, ax=axes, label='Correlation', ticks=[-1, -0.5, 0, 0.5, 1],
             orientation='vertical', fraction=0.02, pad=0.02)

fig.suptitle('Correlation of 500 hPa $\\bar{u}$ and div1 variants', fontsize=16, y=0.985)
save_path = '/home/links/ct715/eddy_feedback/chapter1/reanalysis/saffin-like_plots/plots'
save_file = os.path.join(save_path, f'saffin-like_corrs_all-k_500hPa.png')
plt.savefig(save_file)