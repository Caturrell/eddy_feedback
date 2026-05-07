import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from eofs.standard import Eof
import functions.SIT_functions.SIT_eddy_feedback_functions as eff

# ── Data loading ────────────────────────────────────────────────────────────
jra55_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/daily_uvtw'
jra55_files = sorted([f for f in os.listdir(jra55_dir) if f.endswith('.nc')])
jra55_ds = xr.open_mfdataset(
    [os.path.join(jra55_dir, f) for f in jra55_files], combine='by_coords'
)
u = jra55_ds['u']
u = u.sel(time=slice('1979-01-01', '2014-12-31'))  # ensure consistent time range

press_dim = 'level'   # adjust if JRA55 uses 'lev' or 'pressure'
lat_slice = slice(-20, -80)

# ── Preprocessing ───────────────────────────────────────────────────────────

def deseasonalize(da, time_dim='time'):
    """Remove daily climatology (day-of-year mean)."""
    clim = da.groupby(f'{time_dim}.dayofyear').mean(time_dim)
    return da.groupby(f'{time_dim}.dayofyear') - clim


def linear_detrend(da, time_dim='time'):
    """Remove linear trend along the time axis (reanalysis requirement)."""
    axis = da.dims.index(time_dim)
    detrended = signal.detrend(da.values, axis=axis, type='linear')
    return xr.DataArray(detrended, coords=da.coords, dims=da.dims, attrs=da.attrs)


# Zonal mean → Southern Hemisphere
u_zm = u.mean('lon')
u_south = u_zm.sel(lat=lat_slice)

# Deseasonalise then linearly detrend (following methodology)
u_south = deseasonalize(u_south)
u_south = linear_detrend(u_south)

lats = u_south.lat.values

# ── EOF calculation ──────────────────────────────────────────────────────────

def eof_calc_2d(data, lats):
    """
    Compute EOF of 2-D (time, lat) data following Baldwin et al. (2009).

    Weighting  : sqrt(cos(lat))  — area weighting
    PC scaling : unit variance   — pcscaling=1
    EOF units  : m s⁻¹ PC⁻¹    — via eofsAsCovariance

    Returns
    -------
    eof1 : ndarray (nlat,)   — EOF structure in m s⁻¹ PC⁻¹
    pc1  : ndarray (ntime,)  — SAM index with unit variance
    var_frac : float         — fraction of variance explained
    """
    coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
    wgts = np.sqrt(coslat)[np.newaxis, :]          # (1, nlat)

    solver = Eof(data, weights=wgts, center=True)  # center removes the time-mean

    # eofsAsCovariance + pcscaling=1 → EOF in m s⁻¹ PC⁻¹, PC has unit variance
    eof1 = solver.eofsAsCovariance(neofs=1)[0]     # (nlat,)
    pc1  = solver.pcs(npcs=1, pcscaling=1)[:, 0]   # (ntime,)  unit variance
    var_frac = solver.varianceFraction(neigs=1)[0]

    return eof1, pc1, var_frac


# ── Vertical averaging methods ───────────────────────────────────────────────

def pressure_weighted_mean(da, levels, press_dim):
    """Proper pressure-weighted vertical mean: sum(u * dp) / sum(dp)."""
    p = da[press_dim]
    return da.weighted(p).mean(press_dim)


methods = {
    'Vert. integrated (100–1000 hPa)':      slice(100, 1000),
    'Vert. integrated (250, 500, 850 hPa)': [250., 500., 850.],
}

eof_results = {}
for label, method in methods.items():
    if isinstance(method, slice):
        u_lev = u_south.sel({press_dim: method})
        u_vi  = pressure_weighted_mean(u_lev, u_lev[press_dim], press_dim)

    elif isinstance(method, list):
        u_lev = u_south.sel({press_dim: method}, method='nearest')
        u_vi  = pressure_weighted_mean(u_lev, u_lev[press_dim], press_dim)

    else:                                   # single level
        u_vi = u_south.sel({press_dim: method}, method='nearest')

    data = u_vi.values                      # (time, lat)
    eof1, pc1, var_frac = eof_calc_2d(data, lats)

    eof_results[label] = {'eof1': eof1, 'pc1': pc1, 'var_frac': var_frac}
    print(f'{label}: EOF1 explains {var_frac * 100:.1f}% of variance')
    print(f'  PC std = {pc1.std():.4f}  (should be ~1.0)')
    print(f'  EOF units: m s⁻¹ PC⁻¹,  range [{eof1.min():.3f}, {eof1.max():.3f}]')

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
for label, res in eof_results.items():
    ax.plot(lats, res['eof1'], label=f'{label} ({res["var_frac"] * 100:.1f}%)')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('Latitude')
ax.set_ylabel('EOF1  (m s⁻¹ PC⁻¹)')
ax.set_ylim(-4, 4)
ax.legend()
ax.set_title('SAM EOF1 (1979–2014)')
plt.tight_layout()

save_dir = '/home/links/ct715/eddy_feedback/b-parameter/simpson_2013/jra55_initial_plots/plots'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'sam_eof_comparison.png'), dpi=150)