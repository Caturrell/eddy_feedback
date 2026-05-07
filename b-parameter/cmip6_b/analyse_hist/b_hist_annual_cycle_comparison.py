import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np
import xarray as xr
import os

# ── paths ───────────────────────────────────────────────────────────────────
CSV_PATH = (
    '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/b_annual_cycle_cmip6_hist.csv'
)
REANALYSIS_PATH = (
    '/home/links/ct715/eddy_feedback/b-parameter/reanalysis_b/jra55/jra55_850_sit_plots/'
    '1979_2016/6hourly/level_250_500_850hPa/b_dataset.nc'
)
PLOT_DIR = (
    '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/analyse_hist/b_plots'
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


def get_reanalysis_b_annual_cycle(ds, variant, hemisphere):
    """Return (x, y) annual-cycle arrays for the JRA-55 reanalysis at lag=0."""
    b_arr             = np.full(len(SEASON_LIST), np.nan)
    central_month_arr = np.zeros(len(SEASON_LIST))

    for season in SEASON_LIST:
        var_name = f'ucomp_va_{variant}_va_b_{hemisphere}_{season}'
        if var_name in ds:
            idx                  = CENTRAL_MONTH_DICT[season] - 1
            b_arr[idx]           = float(ds[var_name].mean('lag'))
            central_month_arr[idx] = CENTRAL_MONTH_DICT[season]

    return central_month_arr, b_arr


def get_b_annual_cycle(df, model_name, variant, hemisphere):
    """Return (x, y, tick_labels) annual-cycle arrays for one model/variant/hemisphere."""
    b_arr             = np.full(len(SEASON_LIST), np.nan)
    central_month_arr = np.zeros(len(SEASON_LIST))
    labels            = np.array(['' ] * len(SEASON_LIST))

    sub = df[
        (df['model']      == model_name) &
        (df['variant']    == variant)    &
        (df['hemisphere'] == hemisphere)
    ]
    for _, row in sub.iterrows():
        idx = CENTRAL_MONTH_DICT[row['season']] - 1
        b_arr[idx]             = row['b']
        central_month_arr[idx] = CENTRAL_MONTH_DICT[row['season']]
        labels[idx]            = row['season'][1]

    return central_month_arr, b_arr, labels


def get_multi_model_mean(df, model_names, variant, hemisphere):
    """Return (x, y_mean) multi-model mean annual cycle."""
    all_y = []
    x_ref = None
    for model_name in model_names:
        x, y, _ = get_b_annual_cycle(df, model_name, variant, hemisphere)
        all_y.append(y)
        if x_ref is None:
            x_ref = x
    return x_ref, np.nanmean(all_y, axis=0)


def plot_cmip6_model_comparison(df, model_names, ds_jra):
    """
    6-panel figure: rows = NH / SH, cols = div1_QG / div1_QG_123 / div1_QG_gt3.
    Each line = one CMIP6 historical model, coloured distinctly.
    Thick black line = JRA-55 reanalysis.
    Shared legend placed to the right of all panels.
    """
    n_models = len(model_names)
    cmap     = cm.get_cmap('tab20').resampled(n_models)
    colors   = [cmap(i) for i in range(n_models)]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 7),
        sharey=True,
        #sharex=True,
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

            for model_name, color in zip(model_names, colors):
                x, y, tick_labels = get_b_annual_cycle(df, model_name, variant, hemisphere)
                line, = ax.plot(
                    x, y,
                    color=color, marker='x', markersize=4,
                    linewidth=1.2, label=model_name,
                )
                if row == 0 and col == 0:
                    legend_handles.append(line)
                    legend_labels.append(model_name)

            # ── reanalysis ──────────────────────────────────────────────────
            x_r, y_r = get_reanalysis_b_annual_cycle(ds_jra, variant, hemisphere)
            line_r, = ax.plot(
                x_r, y_r,
                color='black', linewidth=2.5, zorder=5, label='JRA-55',
            )
            if row == 0 and col == 0:
                legend_handles.append(line_r)
                legend_labels.append('JRA-55')

            # ── multi-model mean ─────────────────────────────────────────────
            x_m, y_m = get_multi_model_mean(df, model_names, variant, hemisphere)
            line_m, = ax.plot(
                x_m, y_m,
                color='black', linewidth=2.0, linestyle='--', zorder=4,
                label='Multi-model mean',
            )
            if row == 0 and col == 0:
                legend_handles.append(line_m)
                legend_labels.append('Multi-model mean')

            # ── cosmetics ───────────────────────────────────────────────────
            ax.axhline(0, color='k', linewidth=0.6)
            ax.set_ylim(-0.3, 0.3)
            ax.set_xticks(range(1, len(tick_labels) + 1))
            ax.set_xticklabels(tick_labels, fontsize=8)

            if row == 0:
                ax.set_title(variant, fontsize=10)

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


def plot_cmip6_single_hem(df, model_names, hemisphere, ds_jra):
    """
    3-panel figure for a single hemisphere.
    Each line = one CMIP6 historical model, coloured distinctly.
    Thick black line = JRA-55 reanalysis.
    Shared legend to the right of all panels.
    """
    n_models  = len(model_names)
    cmap      = cm.get_cmap('tab20')
    colors    = [cmap(i) for i in range(n_models)]
    hem_label = HEM_LABELS[hemisphere]

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

        for model_name, color in zip(model_names, colors):
            x, y, tick_labels = get_b_annual_cycle(df, model_name, variant, hemisphere)
            line, = ax.plot(
                x, y,
                color=color, marker='x', markersize=4,
                linewidth=1.2, label=model_name,
            )
            if col == 0:
                legend_handles.append(line)
                legend_labels.append(model_name)

        # ── reanalysis ──────────────────────────────────────────────────────
        x_r, y_r = get_reanalysis_b_annual_cycle(ds_jra, variant, hemisphere)
        line_r, = ax.plot(
            x_r, y_r,
            color='black', linewidth=2.5, zorder=5, label='JRA-55',
        )
        if col == 0:
            legend_handles.append(line_r)
            legend_labels.append('JRA-55')

        # ── multi-model mean ─────────────────────────────────────────────────
        x_m, y_m = get_multi_model_mean(df, model_names, variant, hemisphere)
        line_m, = ax.plot(
            x_m, y_m,
            color='black', linewidth=2.0, linestyle='--', zorder=4,
            label='Multi-model mean',
        )
        if col == 0:
            legend_handles.append(line_m)
            legend_labels.append('Multi-model mean')

        ax.axhline(0, color='k', linewidth=0.6)
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
df = pd.read_csv(CSV_PATH)
model_names = sorted(df['model'].unique())
print(f'Found {len(model_names)} models: {model_names}')

ds_jra = xr.open_dataset(REANALYSIS_PATH)

plot_cmip6_model_comparison(df, model_names, ds_jra)
for hem in HEMISPHERES:
    plot_cmip6_single_hem(df, model_names, hem, ds_jra)
