import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE_DIR = '/gws/ssde/j25a/arctic_connect/cturrell/reanalysis_data/era5/6h_era5/an_ml_extract'
CHECK_YEAR = 1979
CHECK_TIMESTEP = 0  # index into the year file — change to spot-check different times

VARIABLES = ['u', 'v', 't']
LEVELS = [250, 500, 850]

VAR_LABELS = {'u': 'Zonal wind (m s$^{-1}$)', 'v': 'Meridional wind (m s$^{-1}$)', 't': 'Temperature (K)'}
VAR_CMAPS  = {'u': 'RdBu_r', 'v': 'RdBu_r', 't': 'RdYlBu_r'}


def load_snapshot(var):
    path = os.path.join(BASE_DIR, var, f"{CHECK_YEAR}_{var}_6h_snapshots.nc")
    ds = xr.open_dataset(path)
    return ds[var].isel(time=CHECK_TIMESTEP).load()


def main():
    print(f"Loading single timestep (year={CHECK_YEAR}, t={CHECK_TIMESTEP})...")
    data = {var: load_snapshot(var) for var in VARIABLES}
    timestamp = xr.open_dataset(
        os.path.join(BASE_DIR, 'u', f"{CHECK_YEAR}_u_6h_snapshots.nc")
    ).time.isel(time=CHECK_TIMESTEP).values

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.05)
    proj = ccrs.PlateCarree()

    for row, var in enumerate(VARIABLES):
        field = data[var]
        if var in ('u', 'v'):
            abs_max = float(max(abs(field.min()), abs(field.max())))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = float(field.min()), float(field.max())

        for col, lev in enumerate(LEVELS):
            ax = fig.add_subplot(gs[row, col], projection=proj)
            panel = field.sel(level=float(lev))

            im = ax.pcolormesh(
                panel.longitude, panel.latitude, panel.values,
                transform=proj, cmap=VAR_CMAPS[var],
                vmin=vmin, vmax=vmax, rasterized=True,
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            ax.set_global()

            if row == 0:
                ax.set_title(f"{lev} hPa", fontsize=11, fontweight='bold')
            if col == 2:
                cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                                    fraction=0.046, pad=0.04, shrink=0.85)
                cbar.set_label(VAR_LABELS[var], fontsize=8)
                cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"ERA5 snapshot — {str(timestamp)[:16]}", fontsize=13, fontweight='bold', y=0.99)

    out_path = os.path.join(os.path.dirname(__file__), 'check_plots', 'era5_check_uvt.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
