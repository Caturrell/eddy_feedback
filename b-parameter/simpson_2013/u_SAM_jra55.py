import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.stats import pearsonr
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

# ── Save 250/500/850 hPa results ─────────────────────────────────────────────

data_dir = '/home/links/ct715/eddy_feedback/b-parameter/simpson_2013/jra55_initial_plots/data'
os.makedirs(data_dir, exist_ok=True)

res_3lev = eof_results['Vert. integrated (250, 500, 850 hPa)']
save_dict = {
    'eof1': res_3lev['eof1'],
    'pc1': res_3lev['pc1'],
    'var_frac': res_3lev['var_frac'],
    'lats': lats,
}
np.savez(
    os.path.join(data_dir, 'sam_eof_250_500_850_jra55.npz'),
    eof1=res_3lev['eof1'],
    pc1=res_3lev['pc1'],
    var_frac=np.array(res_3lev['var_frac']),
    lats=lats,
)
print(f"Saved 250/500/850 hPa SAM data to {data_dir}/sam_eof_250_500_850_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

times = u_south.time.values

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Pearson correlations between the two methods
vals = list(eof_results.values())
r_eof, p_eof = pearsonr(vals[0]['eof1'], vals[1]['eof1'])
r_pc,  p_pc  = pearsonr(vals[0]['pc1'],  vals[1]['pc1'])

# Top row: EOF1 spatial structure
ax_eof = axes[0]
for label, res in eof_results.items():
    ax_eof.plot(lats, res['eof1'], label=f'{label} ({res["var_frac"] * 100:.1f}%)')
ax_eof.axhline(0, color='k', linewidth=0.5)
ax_eof.set_xlabel('Latitude')
ax_eof.set_ylabel('EOF1  (m s⁻¹ PC⁻¹)')
ax_eof.set_ylim(-4, 4)
ax_eof.legend()
ax_eof.set_title('SAM EOF1 (1979–2014)')
ax_eof.text(0.02, 0.95, f'r = {r_eof:.3f}, p = {p_eof:.2e}',
            transform=ax_eof.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Bottom row: PC1 time series
ax_pc = axes[1]
for label, res in eof_results.items():
    ax_pc.plot(times, res['pc1'], label=label, alpha=0.8)
ax_pc.axhline(0, color='k', linewidth=0.5)
ax_pc.set_xlabel('Time')
ax_pc.set_ylabel('PC1 (normalised)')
ax_pc.legend()
ax_pc.set_title('SAM PC1 (1979–2014)')
ax_pc.text(0.02, 0.95, f'r = {r_pc:.3f}, p = {p_pc:.2e}',
           transform=ax_pc.transAxes, va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()

save_dir = '/home/links/ct715/eddy_feedback/b-parameter/simpson_2013/jra55_initial_plots/plots'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'sam_eof_comparison.png'), dpi=150)