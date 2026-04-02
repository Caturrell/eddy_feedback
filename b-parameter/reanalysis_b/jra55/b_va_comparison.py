import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# ── paths ───────────────────────────────────────────────────────────────────
b_path_full = (
    '/home/links/ct715/eddy_feedback/b-parameter/reanalysis_b/jra55/'
    'jra55_850_sit_plots/1979_2016/6hourly/level_full_100_850/b_dataset.nc'
)
b_path_250_850 = (
    '/home/links/ct715/eddy_feedback/b-parameter/reanalysis_b/jra55/'
    'jra55_850_sit_plots/1979_2016/6hourly/level_250_500_850hPa/b_dataset.nc'
)
b_path_250_850_mean = (
    '/home/links/ct715/eddy_feedback/b-parameter/reanalysis_b/jra55/'
    'jra55_850_sit_plots/1979_2016/6hourly/level_250_500_850hPa_mean/b_dataset.nc'
)
plot_dir_b = '/home/links/ct715/eddy_feedback/b-parameter/reanalysis_b/jra55/b_plots/va_comparison/'
os.makedirs(plot_dir_b, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
VA_STR_DICT = {True: '_va', False: '', 500.: '_500'}
CENTRAL_MONTH_DICT = {
    'DJF': 7, 'JFM': 8, 'FMA': 9, 'MAM': 10,
    'AMJ': 11, 'MJJ': 12, 'JJA': 1, 'JAS': 2,
    'ASO': 3, 'SON': 4, 'OND': 5, 'NDJ': 6,
}
SEASON_LIST   = list(CENTRAL_MONTH_DICT.keys())
DIV1_VARIANTS = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
HEMISPHERES   = ['n', 's']
HEM_LABELS    = {'n': 'NH', 's': 'SH'}

METHOD_INFO = [
    ('full (100–850 hPa)',          '-',  'full'),
    ('250–500–850 hPa (p-weighted)', '--', 'partial'),
    ('250–500–850 hPa (mean)',       ':', 'partial_mean'),
]


def get_b_annual_cycle(b_dataset, variant, hemisphere, va_str):
    """Return (central_month_arr, b_arr, tick_labels) for one variant/hemisphere."""
    b_arr             = np.full(len(SEASON_LIST), np.nan)
    central_month_arr = np.zeros(len(SEASON_LIST))
    labels            = np.array(['' ] * len(SEASON_LIST))

    for season in SEASON_LIST:
        idx        = CENTRAL_MONTH_DICT[season] - 1
        b_var_name = f'ucomp{va_str}_{variant}{va_str}_b_{hemisphere}_{season}'
        if b_var_name in b_dataset:
            b_arr[idx] = float(b_dataset[b_var_name].mean('lag').values)
        central_month_arr[idx] = CENTRAL_MONTH_DICT[season]
        labels[idx]            = season[1]

    return central_month_arr, b_arr, labels


def plot_b_method_comparison(b_datasets, use_va):
    """
    6-panel figure: rows = NH / SH, columns = div1_QG / div1_QG_123 / div1_QG_gt3.
    Solid = full (100–850 hPa), dotted = full (100–800 hPa), dashed = 250–500–850 hPa only.
    """
    va_str = VA_STR_DICT[use_va]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 7),
        sharey=True,
        sharex=True,
    )
    fig.suptitle(
        'b-parameter annual cycle — method comparison (vertically averaged)',
        fontsize=13, y=1.01,
    )

    for row, hemisphere in enumerate(HEMISPHERES):
        for col, (variant, color) in enumerate(zip(DIV1_VARIANTS, colors)):
            ax = axes[row, col]

            for label, ls, key in METHOD_INFO:
                b_ds = b_datasets[key]
                x, y, tick_labels = get_b_annual_cycle(b_ds, variant, hemisphere, va_str)
                ax.plot(x, y, linestyle=ls, color=color, marker='x',
                        label=label, markersize=5)

            # ── cosmetics ───────────────────────────────────────────────────
            ax.axhline(0, color='k', linewidth=0.6)
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(-0.2, 0.2)

            # column titles on top row only
            if row == 0:
                ax.set_title(variant, fontsize=10)

            # x-tick labels on bottom row only (sharex handles suppression)
            if row == 1:
                ax.set_xticks(range(1, len(tick_labels) + 1))
                ax.set_xticklabels(tick_labels)

            # hemisphere label on left column only
            if col == 0:
                ax.set_ylabel(f'{HEM_LABELS[hemisphere]}\nb', fontsize=10)

            # legend on top-right panel only
            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()

    fname = 'b_va_comparison_NH_SH.png'
    fpath = os.path.join(plot_dir_b ,fname)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'Saved → {fpath}')
    plt.close(fig)


def plot_b_method_comparison_single_hem(b_datasets, use_va, hemisphere):
    """
    3-panel figure for a single hemisphere: columns = div1_QG / div1_QG_123 / div1_QG_gt3.
    Solid = full (100–850 hPa), dotted = full (100–800 hPa), dashed = 250–500–850 hPa only.
    """
    va_str   = VA_STR_DICT[use_va]
    hem_label = HEM_LABELS[hemisphere]
    colors   = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4),
        sharey=True,
        sharex=True,
    )
    fig.suptitle(
        f'b-parameter annual cycle — method comparison — {hem_label} (vertically averaged)',
        fontsize=13, y=1.02,
    )

    for col, (variant, color) in enumerate(zip(DIV1_VARIANTS, colors)):
        ax = axes[col]

        for label, ls, key in METHOD_INFO:
            b_ds = b_datasets[key]
            x, y, tick_labels = get_b_annual_cycle(b_ds, variant, hemisphere, va_str)
            ax.plot(x, y, linestyle=ls, color=color, marker='x',
                    label=label, markersize=5)

        ax.axhline(0, color='k', linewidth=0.6)
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-0.2, 0.2)
        ax.set_title(variant, fontsize=10)
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels)

        if col == 0:
            ax.set_ylabel(f'{hem_label}\nb', fontsize=10)
        if col == 2:
            ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()

    fname = f'b_va_comparison_{hem_label}.png'
    fpath = os.path.join(plot_dir_b, fname)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'Saved → {fpath}')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
b_datasets = {
    'full':         xr.open_dataset(b_path_full),
    'partial':      xr.open_dataset(b_path_250_850),
    'partial_mean': xr.open_dataset(b_path_250_850_mean),
}

plot_b_method_comparison(b_datasets, True)
for hem in HEMISPHERES:
    plot_b_method_comparison_single_hem(b_datasets, True, hem)