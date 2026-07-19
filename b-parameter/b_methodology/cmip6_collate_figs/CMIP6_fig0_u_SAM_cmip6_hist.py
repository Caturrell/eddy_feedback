"""
Collates the SAM EOF1 diagnostic (pressure-weighted 250/500/850 hPa ucomp,
southern hemisphere) across all CMIP6 models into a grid figure (one panel
per model) and a spaghetti figure (all models overlaid on one axes, plus the
multi-model mean and the JRA55 reanalysis as thick reference lines).

Only EOF1 itself is plotted here (not PC1).
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from eofs.standard import Eof
import logging
import os

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('xarray').setLevel(logging.WARNING)

# ── paths ─────────────────────────────────────────────────────────────────────
JRA55_NPZ = (
    '/home/users/cturrell/documents/eddy_feedback/b-parameter/simpson_2013/'
    'data/sam_eofs/sam_eof_250_500_850_jra55.npz'
)
CMIP6_BASE     = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
POSSIBLE_SPANS = ['1850_2015', '1850_2014', '1950_2015', '1950_2014']
ANOM_SUBPATH   = '6hrPlevPt/1979_2015/anoms_ucomp.nc'

script_dir = os.path.dirname(os.path.abspath(__file__))

PLOT_DIR = os.path.join(script_dir, 'plots')
# The per-model grid goes in a subdir; the spaghetti figure (this quantity's
# collated summary) is saved directly under PLOT_DIR, matching the other
# CMIP6_* collation scripts.
SUBPLOT_DIR = os.path.join(PLOT_DIR, 'fig0_sam_eof')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SUBPLOT_DIR, exist_ok=True)

# EOF computation is expensive, so per-model results are cached here
# (unrelated to plot output location, so left where it already was).
DATA_DIR = (
    '/home/users/cturrell/documents/eddy_feedback/b-parameter/simpson_2013/'
    'data/sam_eofs'
)
os.makedirs(DATA_DIR, exist_ok=True)

LAT_SLICE        = slice(-80, -20)   # SH (lat coord is S→N, so -80…-20)
LEVELS_HPA       = [250., 500., 850.]
START_YEAR, END_YEAR = 1979, 2015

# Common daily time axis: 1979-01-01 to 2014-12-31 (matches JRA-55 record)
COMMON_TIMES = np.arange(np.datetime64('1979-01-01'), np.datetime64('2015-01-01'))


# ── EOF helpers ───────────────────────────────────────────────────────────────

def eof_calc_2d(data, lats):
    """
    EOF1 of (time, lat) data with sqrt(cos(lat)) weighting.
    Returns (eof1, pc1, var_frac) in covariance scaling (m s⁻¹ PC⁻¹).
    """
    coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
    wgts   = np.sqrt(coslat)[np.newaxis, :]
    solver = Eof(data, weights=wgts, center=True)
    eof1     = solver.eofsAsCovariance(neofs=1)[0]
    pc1      = solver.pcs(npcs=1, pcscaling=1)[:, 0]
    var_frac = float(solver.varianceFraction(neigs=1)[0])
    return eof1, pc1, var_frac


def pressure_weighted_mean(da, press_dim='pfull'):
    """Sum(u * p) / sum(p) over the pressure dimension."""
    p = da.coords[press_dim]
    return da.weighted(p).mean(press_dim)


def apply_sign(eof1, pc1):
    """Flip both EOF and PC so the dominant EOF lobe is positive."""
    sign = 1.0 if eof1[np.nanargmax(np.abs(eof1))] > 0 else -1.0
    return sign * eof1, sign * pc1


# ── data loaders ──────────────────────────────────────────────────────────────

def load_jra55():
    logging.info(f'Reading JRA-55 npz: {JRA55_NPZ}')
    d        = np.load(JRA55_NPZ)
    lats     = np.abs(d['lats'])
    eof1, pc1 = apply_sign(d['eof1'], d['pc1'])
    var_frac = float(d['var_frac'])
    # Reconstruct daily time axis (13149 days: 1979-01-01 → 2014-12-31)
    times = COMMON_TIMES[:len(pc1)]
    logging.info(f'JRA-55: {len(lats)} lat points ({lats.min():.1f}–{lats.max():.1f}°S abs), '
                 f'var_frac={var_frac * 100:.1f}%, EOF1 range=[{eof1.min():.3f}, {eof1.max():.3f}], '
                 f'PC1 length={len(pc1)}')
    return lats, eof1, pc1, times, var_frac


def compute_sam_eof_from_anoms(fpath):
    """
    Load pre-computed ucomp anomalies, pressure-weight 250/500/850 hPa,
    select SH latitudes, and compute EOF1 and PC1.
    Returns (abs_lats, eof1, pc1, times, var_frac).
    """
    logging.info(f'  Opening: {fpath}')
    ds = xr.open_dataset(fpath)

    # Time-filter to the common period
    t0, t1 = ds.time.values[[0, -1]]
    logging.info(f'  Dataset time span: {str(t0)[:10]} → {str(t1)[:10]}, '
                 f'filtering to {START_YEAR}–{END_YEAR}')
    ds = ds.sel(time=slice(str(START_YEAR), str(END_YEAR)))
    logging.info(f'  Time steps after filter: {ds.sizes["time"]}')

    # Select levels and SH latitudes (can't mix method='nearest' with slice)
    logging.info(f'  Selecting pfull={LEVELS_HPA} hPa and lat {LAT_SLICE}')
    da = ds['ucomp_anom'].sel(pfull=LEVELS_HPA, method='nearest').sel(lat=LAT_SLICE)
    times_raw = ds.time.values
    actual_levels = da.pfull.values.tolist()
    logging.info(f'  Actual pfull levels selected: {actual_levels}')
    logging.info(f'  Lat range selected: {float(da.lat.values[0]):.2f} → {float(da.lat.values[-1]):.2f}, '
                 f'n={da.sizes["lat"]}')

    # Load into memory before closing the file
    logging.info('  Loading data into memory …')
    da.load()
    ds.close()

    # Pressure-weighted vertical mean → (time, lat)
    logging.info('  Computing pressure-weighted vertical mean')
    u_va = pressure_weighted_mean(da, press_dim='pfull')

    lats = u_va.lat.values
    data = u_va.values

    # Drop any time steps that are all-NaN
    valid     = ~np.all(np.isnan(data), axis=1)
    n_before  = data.shape[0]
    times_raw = times_raw[valid]
    data      = data[valid]
    n_dropped = n_before - data.shape[0]
    if n_dropped:
        logging.info(f'  Dropped {n_dropped} all-NaN time steps (kept {data.shape[0]})')
    logging.info(f'  Input to EOF: shape={data.shape}, '
                 f'NaN fraction={np.isnan(data).mean():.4f}')

    logging.info('  Computing EOF1 …')
    eof1, pc1, var_frac = eof_calc_2d(data, lats)
    eof1, pc1 = apply_sign(eof1, pc1)
    logging.info(f'  EOF1 done: var_frac={var_frac * 100:.1f}%, '
                 f'range=[{eof1.min():.3f}, {eof1.max():.3f}]')

    return np.abs(lats), eof1, pc1, times_raw, var_frac


def _cache_path(model):
    return os.path.join(DATA_DIR, f'{model}_sam_eof_250_500_850.npz')


def load_cmip6_eofs():
    """Return dict: model_name → (abs_lats, eof1, pc1, times, var_frac), and failed dict."""
    all_models = sorted(os.listdir(CMIP6_BASE))
    logging.info(f'Found {len(all_models)} entries in {CMIP6_BASE}')
    logging.info(f'Cache directory: {DATA_DIR}')

    results = {}
    skipped = []
    failed  = {}   # model → reason string

    for i, model in enumerate(all_models, 1):
        logging.info(f'--- [{i}/{len(all_models)}] {model} ---')

        cache = _cache_path(model)
        if os.path.isfile(cache):
            d = np.load(cache)
            if 'times_int64' not in d or 'pc1' not in d:
                logging.info(f'  Stale cache (missing pc1/times) — recomputing: {cache}')
                os.remove(cache)
            else:
                logging.info(f'  Cache hit — loading from {cache}')
                times_int64 = d['times_int64']
                times = times_int64.astype('datetime64[ns]') if len(times_int64) else None
                results[model] = (d['abs_lats'], d['eof1'], d['pc1'], times, float(d['var_frac']))
                logging.info(f'  Loaded: var_frac={float(d["var_frac"]) * 100:.1f}%, '
                             f'EOF1 range=[{d["eof1"].min():.3f}, {d["eof1"].max():.3f}], '
                             f'PC1 length={len(d["pc1"])}')
                continue

        fpath = None
        for ts in POSSIBLE_SPANS:
            candidate = os.path.join(CMIP6_BASE, model, ts, ANOM_SUBPATH)
            if os.path.isfile(candidate):
                fpath = candidate
                logging.info(f'  No cache — found anoms file under time span: {ts}')
                break

        if fpath is None:
            logging.info('  No anoms_ucomp.nc found and no cache — skipping')
            skipped.append(model)
            continue

        try:
            abs_lats, eof1, pc1, times, vf = compute_sam_eof_from_anoms(fpath)
            try:
                times_int64 = times.astype('datetime64[ns]').astype(np.int64)
            except ValueError:
                logging.warning(f'  Non-standard calendar detected — PC1 time axis will be skipped')
                times_int64 = np.array([], dtype=np.int64)
                times       = None
            results[model] = (abs_lats, eof1, pc1, times, vf)
            np.savez(
                cache,
                abs_lats=abs_lats, eof1=eof1, pc1=pc1,
                times_int64=times_int64,
                var_frac=vf,
            )
            logging.info(f'  ✓ {model}: var_frac={vf * 100:.1f}% — saved to {cache}')
        except Exception as e:
            reason = f'{type(e).__name__}: {e}'
            logging.warning(f'  ✗ {model} FAILED: {reason}')
            failed[model] = reason

    logging.info(f'Summary: {len(results)} loaded, {len(skipped)} skipped (no file), '
                 f'{len(failed)} failed')
    if skipped:
        logging.info(f'  Skipped: {skipped}')
    if failed:
        logging.info(f'  Failed:  {list(failed.keys())}')
    return results, failed


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_sam_eof_grid(cmip6_data):
    """One panel per model: SAM EOF1 vs latitude."""
    used_models = sorted(cmip6_data)
    n_models = len(used_models)
    n_cols = 6
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3. * n_cols, 2.5 * n_rows), sharex=True
    )
    axes = np.atleast_2d(axes)
    fig_title = fig.suptitle(
        r'SAM EOF1 of $[\overline{u}]_s$ (all CMIP6 models)', fontsize=14
    )
    for extra_ax in axes.flat[n_models:]:
        extra_ax.axis('off')

    legend_handles = None
    for ax, model in zip(axes.flat, used_models):
        lats, eof1, _pc1, _times, _vf = cmip6_data[model]
        sort_idx = np.argsort(lats)
        (h,) = ax.plot(lats[sort_idx], eof1[sort_idx], color='blue', label='EOF1')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(model, fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)
        if legend_handles is None:
            legend_handles = [h]

    for row in axes:
        row[0].set_ylabel('EOF1 (m s$^{-1}$ PC$^{-1}$)', fontsize=8)
    active_ids = {id(ax) for ax in axes.flat[:n_models]}
    for col in range(n_cols):
        col_active = [ax for ax in axes[:, col] if id(ax) in active_ids]
        if col_active:
            col_active[-1].set_xlabel('latitude (°S, absolute)', fontsize=8)

    fig.legend(handles=legend_handles, loc='lower center', ncol=1, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.tight_layout(rect=(0., 0.02, 1., 0.97))

    out_file = os.path.join(SUBPLOT_DIR, 'CMIP6_fig0_sam-eof_grid.png')
    fig.savefig(out_file, bbox_extra_artists=(fig_title,), bbox_inches='tight', dpi=150)
    plt.close(fig)
    logging.info(f'Saved {out_file}')


def plot_sam_eof_spaghetti(jra55_lats, jra55_eof, jra55_vf, cmip6_data):
    """All models' EOF1 overlaid, plus the multi-model mean and JRA55."""
    used_models = sorted(cmip6_data)
    n_models = len(used_models)
    model_colors = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, n_models))

    fig, ax = plt.subplots(figsize=(15., 6.5))
    fig_title = fig.suptitle(
        r'SAM EOF1 of $[\overline{u}]_s$ - CMIP6 models vs JRA55', fontsize=14
    )

    for model, color in zip(used_models, model_colors):
        lats, eof1, _pc1, _times, _vf = cmip6_data[model]
        sort_idx = np.argsort(lats)
        ax.plot(lats[sort_idx], eof1[sort_idx], color=color, lw=0.8, alpha=0.7)

    # Multi-model mean, interpolated onto JRA55's latitude grid
    sort_jra = np.argsort(jra55_lats)
    jra_lats_sorted = jra55_lats[sort_jra]
    all_eofs_on_jra_lats = []
    for model in used_models:
        lats, eof1, _pc1, _times, _vf = cmip6_data[model]
        sort_idx = np.argsort(lats)
        all_eofs_on_jra_lats.append(
            np.interp(jra_lats_sorted, lats[sort_idx], eof1[sort_idx])
        )
    mmm_eof = np.nanmean(all_eofs_on_jra_lats, axis=0)
    logging.info(f'  MMM EOF1 range: [{mmm_eof.min():.3f}, {mmm_eof.max():.3f}]')

    (h_mmm,) = ax.plot(jra_lats_sorted, mmm_eof, color='k', lw=2.0, linestyle='--',
                        zorder=4, label='Multi-model mean')
    (h_jra,) = ax.plot(jra_lats_sorted, jra55_eof[sort_jra], color='k', lw=2.5,
                        zorder=5, label=f'JRA55 ({jra55_vf * 100:.1f}%)')

    ax.axhline(0, color='0.3', linewidth=0.6)
    ax.set_xlabel('latitude (°S, absolute)', fontsize=11)
    ax.set_ylabel('EOF1 (m s$^{-1}$ PC$^{-1}$)', fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(True)

    fig.subplots_adjust(right=0.72, top=0.9)

    model_legend_handles = [
        plt.Line2D([0], [0], color=color, lw=1.2, label=model)
        for model, color in zip(used_models, model_colors)
    ]
    model_legend = ax.legend(
        handles=model_legend_handles, loc='upper left', bbox_to_anchor=(1.005, 1.02),
        fontsize=6, ncol=2, title='CMIP6 models', title_fontsize=7, frameon=False
    )
    ax.add_artist(model_legend)
    ax.legend(handles=[h_mmm, h_jra], loc='upper left', fontsize=8, frameon=True)

    out_file = os.path.join(PLOT_DIR, 'CMIP6_fig0_sam-eof_spaghetti.png')
    fig.savefig(out_file, bbox_extra_artists=(model_legend, fig_title),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    logging.info(f'Saved {out_file}')


# ── main ──────────────────────────────────────────────────────────────────────
logging.info('=== SAM EOF1 comparison: CMIP6 historical vs JRA-55 ===')

logging.info('--- Loading JRA-55 ---')
jra55_lats, jra55_eof, _jra55_pc1, _jra55_times, jra55_vf = load_jra55()

logging.info('--- Loading CMIP6 models ---')
cmip6_data, failed_models = load_cmip6_eofs()

logging.info(f'--- Plotting ({len(cmip6_data)} models) ---')
if not cmip6_data:
    logging.error('No CMIP6 models loaded — cannot produce plot.')
else:
    plot_sam_eof_grid(cmip6_data)
    plot_sam_eof_spaghetti(jra55_lats, jra55_eof, jra55_vf, cmip6_data)

if failed_models:
    logging.warning('=== Failed models ===')
    for model, reason in failed_models.items():
        logging.warning(f'  {model}: {reason}')
else:
    logging.info('No models failed.')

logging.info('=== Done ===')
