import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from eofs.standard import Eof
from scipy.stats import pearsonr
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

PLOT_DIR = (
    '/home/users/cturrell/documents/eddy_feedback/b-parameter/simpson_2013/plots'
)
DATA_DIR = (
    '/home/users/cturrell/documents/eddy_feedback/b-parameter/simpson_2013/'
    'data/sam_eofs'
)
os.makedirs(PLOT_DIR, exist_ok=True)
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


def align_pc1_to_common(times, pc1):
    """
    Map a model's PC1 (with its own time axis) onto COMMON_TIMES.
    Days with no model data are left as NaN.
    """
    aligned = np.full(len(COMMON_TIMES), np.nan)
    model_days  = times.astype('datetime64[D]').astype(np.int64)
    common_days = COMMON_TIMES.astype('datetime64[D]').astype(np.int64)
    idx = np.searchsorted(common_days, model_days)
    # clip before indexing to avoid out-of-bounds, then verify exact match
    idx_clipped = np.minimum(idx, len(common_days) - 1)
    mask = (idx < len(common_days)) & (common_days[idx_clipped] == model_days)
    aligned[idx_clipped[mask]] = pc1[mask]
    return aligned


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

def plot_sam_eof_comparison(jra55_lats, jra55_eof, jra55_pc1, jra55_times,
                            jra55_vf, cmip6_data):
    logging.info('Building SAM EOF comparison plot …')
    n_models = len(cmip6_data)
    cmap     = cm.get_cmap('tab20').resampled(n_models)
    colors   = [cmap(i) for i in range(n_models)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        'SAM EOF1 — CMIP6 historical vs JRA-55 (pressure-weighted 250–500–850 hPa, 1979–2015)',
        fontsize=12,
    )
    ax_eof, ax_pc = axes

    legend_handles = []
    legend_labels  = []

    # ── EOF1 panel ────────────────────────────────────────────────────────────
    logging.info(f'  Plotting {n_models} CMIP6 EOF1 lines')
    for (model, (lats, eof1, pc1, times, _)), color in zip(cmip6_data.items(), colors):
        line, = ax_eof.plot(lats, eof1, color=color, linewidth=1.0, alpha=0.4, label=model)
        legend_handles.append(line)
        legend_labels.append(model)

    # Multi-model mean EOF1
    logging.info('  Computing multi-model mean EOF1')
    all_eofs_on_jra_lats = []
    for lats, eof1, pc1, times, _ in cmip6_data.values():
        sort_idx   = np.argsort(lats)
        eof_interp = np.interp(np.sort(jra55_lats), lats[sort_idx], eof1[sort_idx])
        all_eofs_on_jra_lats.append(eof_interp)
    mmm_eof = np.nanmean(all_eofs_on_jra_lats, axis=0)
    logging.info(f'  MMM EOF1 range: [{mmm_eof.min():.3f}, {mmm_eof.max():.3f}]')
    line_m, = ax_eof.plot(
        np.sort(jra55_lats), mmm_eof,
        color='black', linewidth=2.0, linestyle='--', zorder=4, label='Multi-model mean',
    )
    legend_handles.append(line_m)
    legend_labels.append('Multi-model mean')

    # JRA-55 EOF1
    sort_jra = np.argsort(jra55_lats)
    line_r, = ax_eof.plot(
        jra55_lats[sort_jra], jra55_eof[sort_jra],
        color='black', linewidth=2.5, zorder=5, label=f'JRA-55 ({jra55_vf * 100:.1f}%)',
    )
    legend_handles.append(line_r)
    legend_labels.append(f'JRA-55 ({jra55_vf * 100:.1f}%)')

    ax_eof.axhline(0, color='k', linewidth=0.6)
    ax_eof.set_xlabel('Latitude (°S, absolute)', fontsize=10)
    ax_eof.set_ylabel('EOF1  (m s⁻¹ PC⁻¹)', fontsize=10)

    # ── PC1 panel ─────────────────────────────────────────────────────────────
    logging.info('  Plotting PC1 time series')

    # Align all model PC1s to COMMON_TIMES and compute MMM PC1
    all_pc1_aligned = []
    n_skipped_pc1 = 0
    for (model, (lats, eof1, pc1, times, _)), color in zip(cmip6_data.items(), colors):
        if times is None:
            logging.info(f'  Skipping PC1 for {model} (non-standard calendar)')
            n_skipped_pc1 += 1
            continue
        ax_pc.plot(times, pc1, color=color, linewidth=0.8, alpha=0.25, label=model)
        all_pc1_aligned.append(align_pc1_to_common(times, pc1))
    if n_skipped_pc1:
        logging.info(f'  {n_skipped_pc1} model(s) skipped in PC1 panel (non-standard calendar)')

    mmm_pc1 = np.nanmean(all_pc1_aligned, axis=0)
    logging.info(f'  MMM PC1 range: [{np.nanmin(mmm_pc1):.3f}, {np.nanmax(mmm_pc1):.3f}]')

    ax_pc.plot(
        COMMON_TIMES, mmm_pc1,
        color='black', linewidth=1.5, linestyle='--', zorder=4, label='Multi-model mean',
    )

    # JRA-55 PC1
    ax_pc.plot(
        jra55_times, jra55_pc1,
        color='black', linewidth=2.0, zorder=5, label=f'JRA-55 ({jra55_vf * 100:.1f}%)',
    )

    # Correlation: MMM vs JRA-55 on overlapping non-NaN days
    overlap_mask = ~np.isnan(mmm_pc1) & ~np.isnan(jra55_pc1)
    if overlap_mask.sum() > 2:
        r, p = pearsonr(mmm_pc1[overlap_mask], jra55_pc1[overlap_mask])
        logging.info(f'  MMM vs JRA-55 PC1: r={r:.3f}, p={p:.2e} (n={overlap_mask.sum()})')
        ax_pc.text(
            0.02, 0.95,
            f'MMM vs JRA-55:  r = {r:.3f},  p = {p:.2e}',
            transform=ax_pc.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
        )
    else:
        logging.warning('  Not enough overlapping points for MMM vs JRA-55 correlation')

    ax_pc.axhline(0, color='k', linewidth=0.6)
    ax_pc.set_xlabel('Time', fontsize=10)
    ax_pc.set_ylabel('PC1 (normalised)', fontsize=10)

    # ── shared legend ─────────────────────────────────────────────────────────
    fig.legend(
        legend_handles, legend_labels,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7.5,
        frameon=True,
        ncol=1,
        title='Model',
        title_fontsize=9,
    )

    fig.tight_layout()
    fpath = os.path.join(PLOT_DIR, 'sam_eof1_cmip6_hist_vs_jra55.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    logging.info(f'Saved → {fpath}')
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
logging.info('=== SAM EOF1 comparison: CMIP6 historical vs JRA-55 ===')

logging.info('--- Loading JRA-55 ---')
jra55_lats, jra55_eof, jra55_pc1, jra55_times, jra55_vf = load_jra55()

logging.info('--- Loading CMIP6 models ---')
cmip6_data, failed_models = load_cmip6_eofs()

logging.info(f'--- Plotting ({len(cmip6_data)} models) ---')
if not cmip6_data:
    logging.error('No CMIP6 models loaded — cannot produce plot.')
else:
    plot_sam_eof_comparison(
        jra55_lats, jra55_eof, jra55_pc1, jra55_times, jra55_vf, cmip6_data,
    )

if failed_models:
    logging.warning('=== Failed models ===')
    for model, reason in failed_models.items():
        logging.warning(f'  {model}: {reason}')
else:
    logging.info('No models failed.')

logging.info('=== Done ===')
