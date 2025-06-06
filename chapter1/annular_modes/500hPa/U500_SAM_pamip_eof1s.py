import xarray as xr
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from eofs.xarray import Eof
import math
import pandas as pd

import pdb

import functions.data_wrangling as dw
import functions.eddy_feedback as ef

# =============================================================================
# Setup: Data directories and model list
# =============================================================================
data_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC'
path_dir = Path(data_dir)
models = [m.name.split('_')[0] for m in sorted(path_dir.iterdir())]

# Load all datasets and assign a model attribute for later use
ds = {
    model: xr.open_dataset(f'{data_dir}/{model}_1.1_u_ubar_epfy_divFy.nc').assign_attrs(model=model)
    for model in models
}

# =============================================================================
# Functions
# =============================================================================

def eof_calc_alt(data, lats):
    """Compute the EOF using weighted covariance."""
    coslat = np.cos(np.deg2rad(lats.values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[np.newaxis, :, np.newaxis] # colon for latitude variable
    
    solver = Eof(data, weights=wgts, center=True)
    eofs = solver.eofsAsCovariance(neofs=1)
    pc1 = solver.pcs(npcs=1) #, pcscaling=1) <---- removed because pcscaling=1 is unit variance
    variance_fractions = solver.varianceFraction(neigs=3)
    return eofs, pc1, variance_fractions, solver

def compute_anomalies(ds_model, season, months, vertical_average=False):
    """
    Calculate the seasonal anomalies; if season is FY, average over all time,
    otherwise select desired months and average over time. Optionally compute
    a weighted vertical average.
    """
    seasonal_avg = (
        ds_model.mean(dim='time')
        if season == "FY"
        else ds_model.sel(time=ds_model.time.dt.month.isin(months)).mean(dim='time')
    )
    u_ens = seasonal_avg['u']
    # Compute ensemble anomalies and select the desired region.
    var_anoms = u_ens - u_ens.mean(dim='ens_ax')
    var_anoms = var_anoms.sel(lat=slice(-90,0), level=500).rename({'ens_ax': 'time'})
    
    if vertical_average:
        dp = var_anoms.level.diff('level')
        var_anoms = (var_anoms * dp).sum('level') / dp.sum('level')
        var_anoms.name = 'u'
    return var_anoms

def get_or_compute_eof(model, season, months, save_file, vertical_average=False):
    """
    Check if the EOF dataset for a given model & season exists.
    If not, compute it and save the result.
    """
    if os.path.isfile(save_file):
        print(f"Loading existing EOFs for {model} ({season})")
        return xr.open_dataset(save_file)
    else:
        print(f"Computing EOFs for {model} ({season}) and saving to {save_file}")
        ds_model = ds[model]
        # if model == 'HadGEM3-GC31-LL':
        #     ds_model = ds_model.where(ds_model.level < 1000) # needed for SAM script
        
        var_anoms = compute_anomalies(ds_model, season, months, vertical_average)
        eofs, pc1, var_frac, _ = eof_calc_alt(var_anoms, var_anoms.lat)
        
        eofs_ds = xr.Dataset(coords=eofs.coords)
        eofs_ds['eofs'] = eofs
        eofs_ds['pc1'] = pc1
        eofs_ds['variance_fractions'] = var_frac
        eofs_ds.to_netcdf(save_file)
        print(f"EOFs computed and saved for {model} ({season})")
        return xr.open_dataset(save_file)

# =============================================================================
# Main Script: Directories, Seasons, and Processing Loop
# =============================================================================

# Create directories for saving EOF datasets and plots (if needed)
save_data_path = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/500hPa/data/pamip_SAM_U500'
os.makedirs(save_data_path, exist_ok=True)
plot_base_path = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/500hPa/plots/pamip_eof1s_SAM_U500'
os.makedirs(plot_base_path, exist_ok=True)

# Define seasons and corresponding month selections
seasons = {
    "JAS": [7, 8, 9],
    "FY": list(range(1, 13))
}


for season, months in seasons.items():
    print(f"\nProcessing season: {season}")
    plot_save_path = os.path.join(plot_base_path, season)
    os.makedirs(plot_save_path, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 1: Facet Grid of Full Resolution EOF Fields (Polar Projection - SH)
    # -------------------------------------------------------------------------
    results = {}
    for model in models:
        file_name = f'{model}_eofs_{season}.nc'
        data_file = os.path.join(save_data_path, file_name)
        results[model] = get_or_compute_eof(model, season, months, data_file, vertical_average=False)

    n_models = len(models)
    cols = 4
    rows_grid = math.ceil(n_models / cols)
    fig = plt.figure(figsize=(cols * 5, rows_grid * 5))

    for i, model in enumerate(models):
        ax = plt.subplot(rows_grid, cols, i + 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -20], crs=ccrs.PlateCarree())

        # Get EOF and shift
        eofs = results[model]['eofs']
        eofs_shift = (
            eofs
            .assign_coords(lon=((eofs.lon + 180) % 360) - 180)
            .sortby('lon')
        )
        field = eofs_shift.sel(mode=0)

        # Contour plot
        pcm = field.plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=21,
            cmap="coolwarm",
            extend="both",
            add_colorbar=False,
            zorder=1
        )

        # Add coastlines
        ax.coastlines(linewidth=0.8, zorder=3)

        # Title
        vf = results[model]['variance_fractions'][0].values * 100
        ax.set_title(f"{model} ({vf:.1f}%)", fontsize=10)

    # Add a colorbar outside the subplots
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
    fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', aspect=50)

    fig.suptitle(f"EOF 1 of U500 ({season}, SH)", fontsize=14)
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    facet_plot_file = os.path.join(plot_save_path, f"facet_eof1_{season}_SH.pdf")
    fig.savefig(facet_plot_file)
    print(f"Facet grid plot for {season} (SH) saved to {facet_plot_file}")
    plt.close(fig)

    
    # -------------------------------------------------------------------------
    # Figure 2: Combined Line Plot of U500 EOFs
    # -------------------------------------------------------------------------
    # results2 = {}
    # for model in models:
    #     file_name = f'{model}_eofs_u500_{season}.nc'
    #     data_file = os.path.join(save_data_path, file_name)
    #     results2[model] = get_or_compute_eof(model, season, months, data_file)
    
    # fig2, ax2 = plt.subplots(figsize=(11, 6))
    # for model in models:
    #     eof_ds = results2[model]
        
    #     # Flip the EOF sign based on the test at lat=-60                (SAM SCRIPT)
    #     # if eof_ds['eofs'].sel(lat=-60, mode=0, method='nearest') < 0:
    #     #     eof_ds['eofs'] = eof_ds['eofs'] * -1
        
    #     vf = eof_ds['variance_fractions'][0].values * 100
    #     eof_ds['eofs'].sel(mode=0).plot.line(ax=ax2, label=f"{model} ({vf:.1f}%)")
    
    # ax2.axhline(0, color='k', lw=0.5)
    # ax2.set_xlabel("Latitude")
    # ax2.set_ylabel(r"$u$ EOF 1 ($m/s/PC$)")
    # ax2.set_title(f"EOF 1 Weighted Average (All Models) - {season}")
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # fig2.tight_layout()
    # line_plot_file = os.path.join(plot_save_path, f"line_eof1_{season}.pdf")
    # fig2.savefig(line_plot_file, bbox_inches='tight')
    # print(f"Line plot for {season} saved to {line_plot_file}")
    # plt.close(fig2)
    
    # -------------------------------------------------------------------------
    # Figure 3: Facet Grid of PC1 Bar Plots for each Model
    # -------------------------------------------------------------------------
    fig3, axs3 = plt.subplots(rows_grid, cols, figsize=(cols * 5, rows_grid * 4))
    axs3 = axs3.flatten()

    # Store variance info for saving
    variance_info = []

    for i, model in enumerate(models):
        ax = axs3[i]
        pc1_data = results[model]['pc1'].sel(mode=0)
        variance = np.var(pc1_data)
        variance_info.append({'model': model, 'variance': np.round(variance.values, decimals=2)})
        
        colors = ['red' if v >= 0 else 'blue' for v in pc1_data.values]
        width = pc1_data.time[1] - pc1_data.time[0] if pc1_data.time.size > 1 else 1
        ax.bar(pc1_data.time, pc1_data, color=colors, width=width)
        ax.set_xlabel("Time")
        ax.set_ylabel("PC1 Value")
        ax.set_title(f"{model} ($\sigma^2$ = {variance:.2f})")

    # Remove extra subplots
    for ax in axs3[len(models):]:
        fig3.delaxes(ax)

    fig3.tight_layout()
    pc1_plot_file = os.path.join(plot_save_path, f"bar_pc1_{season}.pdf")
    fig3.savefig(pc1_plot_file, bbox_inches='tight')
    print(f"PC1 bar plot for {season} saved to {pc1_plot_file}")
    plt.close(fig3)

    # Save the variance info to CSV
    variance_df = pd.DataFrame(variance_info)
    data_save_path = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/500hPa/data/pamip_misc'
    os.makedirs(data_save_path, exist_ok=True)
    csv_file_path = os.path.join(data_save_path, f"pc1_variance_{season}.csv")
    variance_df.to_csv(csv_file_path, index=False)
    print(f"Saved variance data to {csv_file_path}")
