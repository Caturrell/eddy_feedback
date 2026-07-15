import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.stats import pearsonr
from eofs.standard import Eof

# ── Data loading ────────────────────────────────────────────────────────────
jra55_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/6h_uvtw/daily_averages/QG_epf_uvtw'
jra55_files = sorted([f for f in os.listdir(jra55_dir) if f.endswith('.nc')])
jra55_ds = xr.open_mfdataset(
    [os.path.join(jra55_dir, f) for f in jra55_files], combine='by_coords'
)
u = jra55_ds['u']
u = u.sel(time=slice('1979-01-01', '2014-12-31'))  # ensure consistent time range

press_dim = 'level'   # adjust if JRA55 uses 'lev' or 'pressure'

# Southern Hemisphere mid-high latitude bounds — build the slice to match
# whichever direction the lat coordinate actually runs (ascending or descending)
lat_bounds = (-80, -20)
lat_ascending = u.lat.values[0] < u.lat.values[-1]
lat_slice = slice(*lat_bounds) if lat_ascending else slice(*lat_bounds[::-1])

# Troposphere pressure-level bounds — same direction-agnostic handling
# (JRA55 stores levels descending: 1000 → 1 hPa)
press_bounds = (100, 850)
press_ascending = u[press_dim].values[0] < u[press_dim].values[-1]
press_slice = slice(*press_bounds) if press_ascending else slice(*press_bounds[::-1])

# Jet-core reference latitude used to pin the arbitrary EOF sign (see
# enforce_sam_sign below) to the Marshall/Thompson-Wallace SAM convention
JET_LAT = -60.

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
    EOF units  : m/s             — via eofsAsCovariance (regression onto standardized PC1)

    Returns
    -------
    eof1 : ndarray (nlat,)   — EOF structure in m/s
    pc1  : ndarray (ntime,)  — SAM index with unit variance
    var_frac : float         — fraction of variance explained
    """
    coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
    wgts = np.sqrt(coslat)[np.newaxis, :]          # (1, nlat)

    solver = Eof(data, weights=wgts, center=True)  # center removes the time-mean

    # eofsAsCovariance + pcscaling=1 → EOF in m/s, PC has unit variance
    eof1 = solver.eofsAsCovariance(neofs=1)[0]     # (nlat,)
    pc1  = solver.pcs(npcs=1, pcscaling=1)[:, 0]   # (ntime,)  unit variance
    var_frac = solver.varianceFraction(neigs=1)[0]

    return eof1, pc1, var_frac


def eof_calc_3d(data, lats, levels):
    """
    Compute EOF of 3-D (time, level, lat) data.

    Weighting  : sqrt(cos(lat)) x sqrt(dp)  — area x mass weighting
    PC scaling : unit variance   — pcscaling=1
    EOF units  : m/s                        — via eofsAsCovariance (regression onto standardized PC1)

    Returns
    -------
    eof1 : ndarray (nlev, nlat) — EOF structure in m/s
    pc1  : ndarray (ntime,)     — SAM index with unit variance
    var_frac : float            — fraction of variance explained
    """
    coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
    lat_wgts = np.sqrt(coslat)                                 # (nlat,)

    dp = np.abs(np.gradient(levels))                           # layer thickness (nlev,)
    lev_wgts = np.sqrt(dp)                                     # (nlev,)

    wgts = lev_wgts[:, np.newaxis] * lat_wgts[np.newaxis, :]   # (nlev, nlat)

    solver = Eof(data, weights=wgts, center=True)

    eof1 = solver.eofsAsCovariance(neofs=1)[0]      # (nlev, nlat)
    pc1  = solver.pcs(npcs=1, pcscaling=1)[:, 0]    # (ntime,)  unit variance
    var_frac = solver.varianceFraction(neigs=1)[0]

    return eof1, pc1, var_frac


def enforce_sam_sign(eof1, pc1, lats, jet_lat=JET_LAT):
    """
    Fix the arbitrary EOF sign so that positive PC1 = positive SAM phase
    (Marshall/Thompson-Wallace convention): EOF1 positive at the jet core.

    Accepts a 1-D (lat,) or 2-D (level, lat) eof1 field.
    """
    jet_idx = np.argmin(np.abs(lats - jet_lat))
    jet_profile = np.atleast_1d(eof1[..., jet_idx])
    ref_val = jet_profile[np.argmax(np.abs(jet_profile))]
    if ref_val < 0:
        eof1 = -eof1
        pc1 = -pc1
    return eof1, pc1


# ── Vertical averaging methods ───────────────────────────────────────────────

def pressure_weighted_mean(da, levels, press_dim):
    """Proper pressure-weighted vertical mean: sum(u * dp) / sum(dp)."""
    p = da[press_dim]
    return da.weighted(p).mean(press_dim)


# Colors are pinned explicitly (rather than left to matplotlib's default
# cycle) so that the vertically-integrated (100-850 hPa) line is the same
# color in panels (b) and (c), and PC1 from the 3-D EOF in panel (a) gets
# its own distinct color in panel (c).
VERT_INT_COLOR    = 'C1'  # Vert. integrated (100-850 hPa) -- shown in (b) and (c)
THREE_LEV_COLOR   = 'C2'  # Vert. integrated (250, 500, 850 hPa) -- shown in (b)
PC1_NOVERT_COLOR  = 'C0'  # No vertical average (pressure-lat EOF, panel a) -- shown in (c)

methods = {
    'Vert. integrated (100–850 hPa)':       (press_slice, VERT_INT_COLOR),
    'Vert. integrated (250, 500, 850 hPa)': ([250., 500., 850.], THREE_LEV_COLOR),
}

eof_results = {}
for label, (method, color) in methods.items():
    if isinstance(method, slice):
        u_lev = u_south.sel({press_dim: method})
        u_vi  = pressure_weighted_mean(u_lev, u_lev[press_dim], press_dim)
        n_levels = u_lev[press_dim].size

    elif isinstance(method, list):
        u_lev = u_south.sel({press_dim: method}, method='nearest')
        u_vi  = pressure_weighted_mean(u_lev, u_lev[press_dim], press_dim)
        n_levels = u_lev[press_dim].size

    else:                                   # single level
        u_vi = u_south.sel({press_dim: method}, method='nearest')
        n_levels = 1

    data = u_vi.values                      # (time, lat)
    eof1, pc1, var_frac = eof_calc_2d(data, lats)
    eof1, pc1 = enforce_sam_sign(eof1, pc1, lats)

    eof_results[label] = {'eof1': eof1, 'pc1': pc1, 'var_frac': var_frac,
                           'n_levels': n_levels, 'color': color}
    print(f'{label}: EOF1 explains {var_frac * 100:.1f}% of variance')
    print(f'  PC std = {pc1.std():.4f}  (should be ~1.0)')
    print(f'  Levels integrated: {n_levels}')
    print(f'  EOF units: m/s,  range [{eof1.min():.3f}, {eof1.max():.3f}]')

# ── 3-D (pressure, time, lat) EOF ────────────────────────────────────────────

u_3d = u_south.sel({press_dim: press_slice}).transpose('time', press_dim, 'lat')
levels_3d = u_3d[press_dim].values
data_3d = u_3d.values                           # (time, nlev, nlat)

eof1_3d, pc1_3d, var_frac_3d = eof_calc_3d(data_3d, lats, levels_3d)
eof1_3d, pc1_3d = enforce_sam_sign(eof1_3d, pc1_3d, lats)

print(f'3D EOF (pressure-lat): explains {var_frac_3d * 100:.1f}% of variance')
print(f'  PC std = {pc1_3d.std():.4f}  (should be ~1.0)')

# ── Save 250/500/850 hPa results ─────────────────────────────────────────────

data_dir = '/home/links/ct715/eddy_feedback/b-parameter/b_methodology/data'
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

# Pearson correlation between the two vertically-averaged methods
vals = list(eof_results.values())
r_eof, _ = pearsonr(vals[0]['eof1'], vals[1]['eof1'])

# PC1 timeseries: 3-D EOF (no vertical average) vs. 100–850 hPa vertical average
pc1_novert = pc1_3d
pc1_vert   = eof_results['Vert. integrated (100–850 hPa)']['pc1']
times      = u_south['time'].values

r_pc1, p_pc1 = pearsonr(pc1_novert, pc1_vert)
print(f'PC1 correlation (no vert. average vs. 100-850 hPa vert. average): '
      f'r = {r_pc1:.3f}, p = {p_pc1:.2e}')


def add_panel_label(ax, label):
    ax.text(0.02, 0.98, f'({label})', transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')


def plot_eof_map(fig, ax, panel_label=None):
    """LHS panel: pressure-latitude EOF1 structure."""
    vmax = np.nanmax(np.abs(eof1_3d))
    contour_levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(lats, levels_3d, eof1_3d, levels=contour_levels,
                      cmap='RdBu_r', extend='both')
    ax.contour(lats, levels_3d, eof1_3d, levels=contour_levels,
               colors='k', linewidths=0.3)
    ax.set_ylim(850, 100)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title(f'SAM EOF1 pressure-latitude structure ({var_frac_3d * 100:.1f}%)')
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('EOF1  (m/s)')
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def plot_eof_lines(ax, panel_label=None):
    """RHS panel: vertically-averaged EOF1 spatial structure comparison."""
    for label, res in eof_results.items():
        ax.plot(lats, res['eof1'], color=res['color'],
                label=f'{label} ({res["var_frac"] * 100:.1f}%, {res["n_levels"]} levels)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('EOF1  (m/s)')
    ax.set_ylim(-4, 4)
    ax.legend()
    ax.set_title('SAM EOF1 (1979–2014)')
    ax.text(0.98, 0.95, f'r = {r_eof:.3f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def plot_pc1_series(ax, panel_label=None):
    """Bottom panel: PC1 timeseries comparison."""
    ax.plot(times, pc1_novert, color=PC1_NOVERT_COLOR,
            label='No vertical average (pressure-lat EOF)', linewidth=0.8)
    ax.plot(times, pc1_vert, color=VERT_INT_COLOR,
            label='Vert. integrated (100–850 hPa)', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('PC1 (unit variance)')
    ax.legend()
    ax.set_title('SAM PC1 timeseries (1979–2014)')
    ax.text(0.98, 0.95, f'r = {r_pc1:.3f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    if panel_label is not None:
        add_panel_label(ax, panel_label)


save_dir = '/home/links/ct715/eddy_feedback/b-parameter/b_methodology/plots'
os.makedirs(save_dir, exist_ok=True)

# Figure 1: EOF1 map + spatial comparison
fig1, (ax_map, ax_eof) = plt.subplots(1, 2, figsize=(14, 5))
plot_eof_map(fig1, ax_map)
plot_eof_lines(ax_eof)
fig1.tight_layout()
fig1.savefig(os.path.join(save_dir, 'sam_eof_comparison.png'), dpi=150)

# Figure 2: PC1 timeseries comparison
fig2, ax_pc = plt.subplots(1, 1, figsize=(10, 4))
plot_pc1_series(ax_pc)
fig2.tight_layout()
fig2.savefig(os.path.join(save_dir, 'pc1_timeseries_comparison.png'), dpi=150)

# Figure 3: combined — EOF1 map + spatial comparison on top row, PC1
# timeseries spanning the full width of the bottom row
fig3 = plt.figure(figsize=(14, 9))
gs = fig3.add_gridspec(2, 2, height_ratios=[1, 0.7])
ax_map3 = fig3.add_subplot(gs[0, 0])
ax_eof3 = fig3.add_subplot(gs[0, 1])
ax_pc3  = fig3.add_subplot(gs[1, :])
plot_eof_map(fig3, ax_map3, panel_label='a')
plot_eof_lines(ax_eof3, panel_label='b')
plot_pc1_series(ax_pc3, panel_label='c')
fig3.tight_layout()
fig3.savefig(os.path.join(save_dir, 'fig1_sam_eof_combined.png'), dpi=150)