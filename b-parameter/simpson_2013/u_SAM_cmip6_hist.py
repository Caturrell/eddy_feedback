import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
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
    'jra55_initial_plots/data/sam_eof_250_500_850_jra55.npz'
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

LAT_SLICE      = slice(-80, -20)   # SH (lat coord is S→N, so -80…-20)
LEVELS_HPA     = [250., 500., 850.]
START_YEAR, END_YEAR = 1979, 2015


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


def sign_correct(eof):
    """Ensure the dominant lobe is positive."""
    idx = np.nanargmax(np.abs(eof))
    return eof if eof[idx] > 0 else -eof


# ── data loaders ──────────────────────────────────────────────────────────────

def load_jra55():
    logging.info(f'Reading JRA-55 npz: {JRA55_NPZ}')
    d        = np.load(JRA55_NPZ)
    lats     = np.abs(d['lats'])
    eof1     = sign_correct(d['eof1'])
    var_frac = float(d['var_frac'])
    logging.info(f'JRA-55: {len(lats)} lat points ({lats.min():.1f}–{lats.max():.1f}°S abs), '
                 f'var_frac={var_frac * 100:.1f}%, EOF1 range=[{eof1.min():.3f}, {eof1.max():.3f}]')
    return lats, eof1, var_frac


def compute_sam_eof_from_anoms(fpath):
    """
    Load pre-computed ucomp anomalies, pressure-weight 250/500/850 hPa,
    select SH latitudes, and compute EOF1.
    Returns (abs_lats, eof1, var_frac).
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
    data      = data[valid]
    n_dropped = n_before - data.shape[0]
    if n_dropped:
        logging.info(f'  Dropped {n_dropped} all-NaN time steps (kept {data.shape[0]})')
    logging.info(f'  Input to EOF: shape={data.shape}, '
                 f'NaN fraction={np.isnan(data).mean():.4f}')

    logging.info('  Computing EOF1 …')
    eof1, _, var_frac = eof_calc_2d(data, lats)
    eof1 = sign_correct(eof1)
    logging.info(f'  EOF1 done: var_frac={var_frac * 100:.1f}%, '
                 f'range=[{eof1.min():.3f}, {eof1.max():.3f}]')

    return np.abs(lats), eof1, var_frac


def _cache_path(model):
    return os.path.join(DATA_DIR, f'{model}_sam_eof_250_500_850.npz')


def load_cmip6_eofs():
    """Return dict: model_name → (abs_lats, eof1, var_frac), and failed dict."""
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
            logging.info(f'  Cache hit — loading from {cache}')
            d = np.load(cache)
            results[model] = (d['abs_lats'], d['eof1'], float(d['var_frac']))
            logging.info(f'  Loaded: var_frac={float(d["var_frac"]) * 100:.1f}%, '
                         f'EOF1 range=[{d["eof1"].min():.3f}, {d["eof1"].max():.3f}]')
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
            abs_lats, eof1, vf = compute_sam_eof_from_anoms(fpath)
            results[model] = (abs_lats, eof1, vf)
            np.savez(cache, abs_lats=abs_lats, eof1=eof1, var_frac=vf)
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

def plot_sam_eof_comparison(jra55_lats, jra55_eof, jra55_vf, cmip6_data):
    logging.info('Building SAM EOF comparison plot …')
    n_models = len(cmip6_data)
    cmap     = cm.get_cmap('tab20').resampled(n_models)
    colors   = [cmap(i) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        'SAM EOF1 — CMIP6 historical vs JRA-55 (pressure-weighted 250–500–850 hPa, 1979–2015)',
        fontsize=12,
    )

    legend_handles = []
    legend_labels  = []

    # CMIP6 models
    logging.info(f'  Plotting {n_models} CMIP6 model lines')
    for (model, (lats, eof1, _)), color in zip(cmip6_data.items(), colors):
        line, = ax.plot(lats, eof1, color=color, linewidth=1.2, label=model)
        legend_handles.append(line)
        legend_labels.append(model)

    # Multi-model mean (interpolate all to JRA55 lats for consistent averaging)
    logging.info('  Computing multi-model mean (interpolated to JRA-55 lats)')
    all_eofs_on_jra_lats = []
    for lats, eof1, _ in cmip6_data.values():
        sort_idx   = np.argsort(lats)
        eof_interp = np.interp(
            np.sort(jra55_lats),
            lats[sort_idx], eof1[sort_idx],
        )
        all_eofs_on_jra_lats.append(eof_interp)
    mmm = np.nanmean(all_eofs_on_jra_lats, axis=0)
    logging.info(f'  Multi-model mean range: [{mmm.min():.3f}, {mmm.max():.3f}]')
    line_m, = ax.plot(
        np.sort(jra55_lats), mmm,
        color='black', linewidth=2.0, linestyle='--', zorder=4,
        label='Multi-model mean',
    )
    legend_handles.append(line_m)
    legend_labels.append('Multi-model mean')

    # JRA-55
    logging.info('  Plotting JRA-55')
    sort_jra = np.argsort(jra55_lats)
    line_r, = ax.plot(
        jra55_lats[sort_jra], jra55_eof[sort_jra],
        color='black', linewidth=2.5, zorder=5,
        label=f'JRA-55 ({jra55_vf * 100:.1f}%)',
    )
    legend_handles.append(line_r)
    legend_labels.append(f'JRA-55 ({jra55_vf * 100:.1f}%)')

    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_xlabel('Latitude (°S, absolute)', fontsize=10)
    ax.set_ylabel('EOF1  (m s⁻¹ PC⁻¹)', fontsize=10)

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
jra55_lats, jra55_eof, jra55_vf = load_jra55()

logging.info('--- Loading CMIP6 models ---')
cmip6_data, failed_models = load_cmip6_eofs()

logging.info(f'--- Plotting ({len(cmip6_data)} models) ---')
if not cmip6_data:
    logging.error('No CMIP6 models loaded — cannot produce plot.')
else:
    plot_sam_eof_comparison(jra55_lats, jra55_eof, jra55_vf, cmip6_data)

if failed_models:
    logging.warning('=== Failed models ===')
    for model, reason in failed_models.items():
        logging.warning(f'  {model}: {reason}')
else:
    logging.info('No models failed.')

logging.info('=== Done ===')
