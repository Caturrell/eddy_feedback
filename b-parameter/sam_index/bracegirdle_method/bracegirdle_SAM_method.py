import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from eofs.standard import Eof

# ── 1. Load data ──────────────────────────────────────────────────────────────
# Bracegirdle method uses geopotential height (Z), not zonal wind
jra55_dir = '/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/hgt_daily'
jra55_files = sorted([f for f in os.listdir(jra55_dir) if f.endswith('.nc')])
jra55_ds = xr.open_mfdataset(
    [os.path.join(jra55_dir, f) for f in jra55_files],
    combine='by_coords'
)

# Adjust variable name to whatever JRA-55 uses for geopotential height
# Common options: 'z', 'hgt', 'zg'
Z = jra55_ds['hgt']

# ── 2. Prepare field: zonal mean, 500 hPa, south of 20°S ─────────────────────
press_dim = 'level'
lat_slice = slice(-90, -20)  # data_checker1000 reorders lat S→N so slice ascending

Z_zm   = Z.mean('lon')                                    # zonal mean
Z_500  = Z_zm.sel({press_dim: 500.}, method='nearest')   # 500 hPa
Z_south = Z_500.sel(lat=lat_slice)                        # poleward of 20°S

lats = Z_south.lat.values
time = Z_south.time.values
ntime = len(time)
nlat  = len(lats)

# ── 3. Deseasonalise (remove daily climatology) ───────────────────────────────
doy = Z_south.time.dt.dayofyear
clim = Z_south.groupby('time.dayofyear').mean('time')     # daily climatology
Z_anom = Z_south.groupby('time.dayofyear') - clim         # anomalies

# ── 4. Detrend (linear) ───────────────────────────────────────────────────────
p = Z_anom.polyfit('time', 1)
trend = xr.polyval(Z_anom['time'], p.polyfit_coefficients)
Z_detrend = Z_anom - trend

# ── 5. Remove global (area-weighted) mean at each timestep ───────────────────
coslat_full = np.cos(np.deg2rad(lats)).clip(0., 1.)
weights_1d  = coslat_full / coslat_full.sum()             # normalised area weights

# Area-weighted mean across domain at each time step
domain_mean = (Z_detrend * weights_1d[np.newaxis, :]).sum('lat')  # (time,)
Z_detrend   = Z_detrend - domain_mean                     # remove global mean

# ── 6. EOF analysis with sqrt(cos(lat)) weighting ────────────────────────────
coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
wgts   = np.sqrt(coslat)[np.newaxis, :]                   # (1, nlat)

data   = Z_detrend.values                                 # (time, lat)
solver = Eof(data, weights=wgts, center=True)

eof1     = solver.eofsAsCovariance(neofs=1)[0]            # (nlat,)
pc1      = solver.pcs(npcs=1, pcscaling=1)[:, 0]         # (time,) — unit variance
var_frac = solver.varianceFraction(neigs=1)

print(f'EOF1 explains {var_frac[0]*100:.1f}% of variance')

# ── 7. E-folding timescale — 180-day window, Gaussian smoothing ───────────────
# Gaussian filter: FWHM = 60 days → sigma = FWHM / (2 * sqrt(2 * ln(2)))
FWHM_days = 60
sigma     = FWHM_days / (2 * np.sqrt(2 * np.log(2)))     # ≈ 25.5 days
max_lag   = 90                                            # lags 0…90 days

def efold_timescale(ac, lags):
    """Find e-folding timescale by linear interpolation where ac crosses 1/e."""
    target = 1.0 / np.e
    for i in range(len(ac) - 1):
        if ac[i] >= target > ac[i + 1]:
            # Linear interpolation
            frac = (ac[i] - target) / (ac[i] - ac[i + 1])
            return lags[i] + frac * (lags[i + 1] - lags[i])
    return np.nan  # autocorrelation never drops to 1/e within window

window     = 180  # days either side of each calendar day
half_win   = window // 2
lags       = np.arange(0, max_lag + 1)
tau_by_day = np.full(365, np.nan)

for doy_i in range(1, 366):
    # Collect all time indices within ±90 days of this day-of-year
    doys_arr = Z_south.time.dt.dayofyear.values
    idx = np.where(
        np.abs(((doys_arr - doy_i + 182) % 365) - 182) <= half_win
    )[0]

    if len(idx) < max_lag + 50:   # skip if insufficient data
        continue

    pc_window = pc1[idx]

    # Smooth with Gaussian filter before computing autocorrelation
    pc_smooth = gaussian_filter1d(pc_window, sigma=sigma)

    # Lagged autocorrelation
    ac = np.array([
        np.corrcoef(pc_smooth[:len(pc_smooth)-lag],
                    pc_smooth[lag:])[0, 1]
        if lag > 0 else 1.0
        for lag in lags
    ])

    tau_by_day[doy_i - 1] = efold_timescale(ac, lags)

# Annual mean timescale
tau_annual = np.nanmean(tau_by_day)
print(f'Annual mean e-folding timescale: {tau_annual:.1f} days')

# ── 8. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# EOF1 spatial pattern
axes[0].plot(lats, eof1)
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].set_xlabel('Latitude')
axes[0].set_ylabel('EOF1 amplitude (covariance)')
axes[0].set_title(f'SAM EOF1 — 500 hPa Z\n({var_frac[0]*100:.1f}% variance)')

# Seasonal cycle of τ
doy_axis = np.arange(1, 366)
axes[1].plot(doy_axis, tau_by_day, color='steelblue')
axes[1].axhline(tau_annual, color='k', linestyle='--',
                label=f'Annual mean = {tau_annual:.1f} d')
axes[1].set_xlabel('Day of year')
axes[1].set_ylabel('E-folding timescale (days)')
axes[1].set_title('SAM decorrelation timescale — seasonal cycle')
axes[1].legend()

plt.tight_layout()
os.makedirs('/home/links/ct715/eddy_feedback/b-parameter/sam_index/jra55_check/plots', exist_ok=True)
plt.savefig('/home/links/ct715/eddy_feedback/b-parameter/sam_index/jra55_check/plots/sam_bracegirdle_efold.png', dpi=150)