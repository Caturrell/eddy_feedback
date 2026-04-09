import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import glob

# ── paths ───────────────────────────────────────────────────────────────────
DATA_ROOT = (
    '/home/users/cturrell/documents/eddy_feedback/'
    'b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015'
)
PLOT_DIR = (
    '/home/users/cturrell/documents/eddy_feedback/'
    'b-parameter/cmip6_b/analyse_hist/b_plots'
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
CENTRAL_MONTH_DICT = {
    'DJF': 7, 'JFM': 8, 'FMA': 9, 'MAM': 10,
    'AMJ': 11, 'MJJ': 12, 'JJA': 1, 'JAS': 2,
    'ASO': 3, 'SON': 4, 'OND': 5, 'NDJ': 6,
}
SEASON_LIST   = list(CENTRAL_MONTH_DICT.keys())
DIV1_VARIANTS = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
HEMISPHERES   = ['n', 's']
HEM_LABELS    = {'n': 'Northern Hemisphere', 's': 'Southern Hemisphere'}
VA_STR        = '_va'


def discover_models(data_root):
    """Return sorted list of (model_name, dataset_path) for all available models."""
    models = []
    for model_dir in sorted(glob.glob(os.path.join(data_root, '*'))):
        nc_path = os.path.join(model_dir, '6hrPlevPt', 'b_dataset.nc')
        if os.path.isfile(nc_path):
            models.append((os.path.basename(model_dir), nc_path))
    return models


def get_b_annual_cycle(b_dataset, variant, hemisphere):
    """Return (x, y, tick_labels) annual-cycle arrays for one variant/hemisphere."""
    b_arr             = np.full(len(SEASON_LIST), np.nan)
    central_month_arr = np.zeros(len(SEASON_LIST))
    labels            = np.array(['' ] * len(SEASON_LIST))

    for season in SEASON_LIST:
        idx        = CENTRAL_MONTH_DICT[season] - 1
        b_var_name = f'ucomp{VA_STR}_{variant}{VA_STR}_b_{hemisphere}_{season}'
        if b_var_name in b_dataset:
            b_arr[idx] = float(b_dataset[b_var_name].mean('lag').values)
        central_month_arr[idx] = CENTRAL_MONTH_DICT[season]
        labels[idx]            = season[1]

    return central_month_arr, b_arr, labels


def plot_cmip6_model_comparison(models, b_datasets):
    """
    6-panel figure: rows = NH / SH, cols = div1_QG / div1_QG_123 / div1_QG_gt3.
    Each line = one CMIP6 historical model, coloured distinctly.
    Shared legend placed to the right of all panels.
    """
    n_models = len(models)
    cmap     = cm.get_cmap('tab20', n_models)
    colors   = [cmap(i) for i in range(n_models)]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 7),
        sharey=True,
        sharex=True,
    )
    fig.suptitle(
        'b-parameter annual cycle — CMIP6 historical models (250–500–850 hPa)',
        fontsize=13,
    )

    legend_handles = []
    legend_labels  = []

    for row, hemisphere in enumerate(HEMISPHERES):
        for col, variant in enumerate(DIV1_VARIANTS):
            ax = axes[row, col]

            for (model_name, _), color in zip(models, colors):
                b_ds = b_datasets[model_name]
                x, y, tick_labels = get_b_annual_cycle(b_ds, variant, hemisphere)
                line, = ax.plot(
                    x, y,
                    color=color, marker='x', markersize=4,
                    linewidth=1.2, label=model_name,
                )
                # collect handles once (from the first panel)
                if row == 0 and col == 0:
                    legend_handles.append(line)
                    legend_labels.append(model_name)

            # ── cosmetics ───────────────────────────────────────────────────
            ax.axhline(0, color='k', linewidth=0.6)
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(-0.3, 0.3)

            if row == 0:
                ax.set_title(variant, fontsize=10)

            if row == 1:
                ax.set_xticks(range(1, len(tick_labels) + 1))
                ax.set_xticklabels(tick_labels, fontsize=8)

            if col == 0:
                ax.set_ylabel(f'$\\bf{{{HEM_LABELS[hemisphere]}}}$\nb', fontsize=10)

    # ── shared legend on the right of all panels ─────────────────────────────
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

    fname = 'b_va_cmip6_hist_annual_cycle_model_comparison.png'
    fpath = os.path.join(PLOT_DIR, fname)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'Saved → {fpath}')
    plt.close(fig)


def plot_cmip6_single_hem(models, b_datasets, hemisphere):
    """
    3-panel figure for a single hemisphere.
    Each line = one CMIP6 historical model, coloured distinctly.
    Shared legend to the right of all panels.
    """
    n_models   = len(models)
    cmap       = cm.get_cmap('tab20', n_models)
    colors     = [cmap(i) for i in range(n_models)]
    hem_label  = HEM_LABELS[hemisphere]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 4),
        sharey=True,
        sharex=True,
    )
    fig.suptitle(
        f'b-parameter annual cycle — CMIP6 historical models — {hem_label} (250–500–850 hPa)',
        fontsize=13,
    )

    legend_handles = []
    legend_labels  = []

    for col, variant in enumerate(DIV1_VARIANTS):
        ax = axes[col]

        for (model_name, _), color in zip(models, colors):
            b_ds = b_datasets[model_name]
            x, y, tick_labels = get_b_annual_cycle(b_ds, variant, hemisphere)
            line, = ax.plot(
                x, y,
                color=color, marker='x', markersize=4,
                linewidth=1.2, label=model_name,
            )
            if col == 0:
                legend_handles.append(line)
                legend_labels.append(model_name)

        ax.axhline(0, color='k', linewidth=0.6)
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-0.3, 0.3)
        ax.set_title(variant, fontsize=10)
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=8)

        if col == 0:
            ax.set_ylabel('b', fontsize=10)

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

    short = 'NH' if hemisphere == 'n' else 'SH'
    fname = f'b_va_cmip6_hist_annual_cycle_{short}.png'
    fpath = os.path.join(PLOT_DIR, fname)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'Saved → {fpath}')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
models = discover_models(DATA_ROOT)
print(f'Found {len(models)} models: {[m for m, _ in models]}')

b_datasets = {name: xr.open_dataset(path) for name, path in models}

plot_cmip6_model_comparison(models, b_datasets)
for hem in HEMISPHERES:
    plot_cmip6_single_hem(models, b_datasets, hem)
