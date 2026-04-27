import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# ── paths ───────────────────────────────────────────────────────────────────
_ERA5_BASE = (
    '/home/users/cturrell/documents/eddy_feedback/b-parameter/'
    'reanalysis_b/era5/era5_sit_plots/1979_2016/6hourly'
)
b_path_250_850 = f'{_ERA5_BASE}/level_250_500_850hPa/b_dataset.nc'
plot_dir_b = (
    '/home/users/cturrell/documents/eddy_feedback/b-parameter/'
    'reanalysis_b/era5/b_method_comparison/b_plots/'
)
os.makedirs(plot_dir_b, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
CENTRAL_MONTH_DICT = {
    'DJF': 7, 'JFM': 8, 'FMA': 9, 'MAM': 10,
    'AMJ': 11, 'MJJ': 12, 'JJA': 1, 'JAS': 2,
    'ASO': 3, 'SON': 4, 'OND': 5, 'NDJ': 6,
}
SEASON_LIST   = list(CENTRAL_MONTH_DICT.keys())
DIV1_VARIANTS = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
HEMISPHERES   = ['n', 's']
HEM_LABELS    = {'n': 'NH', 's': 'SH'}

# label, linestyle, va_str
METHOD_INFO = [
    ('no va', '-',  ''),
    ('va',    '--', '_va'),
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


def plot_b_method_comparison(b_dataset):
    """
    6-panel figure: rows = NH / SH, columns = div1_QG / div1_QG_123 / div1_QG_gt3.
    Solid = no va, dashed = va, dotted = 500 hPa.
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 7),
        sharey=True,
        sharex=True,
    )
    fig.suptitle(
        'b-parameter annual cycle (250–500–850 hPa) — va method comparison',
        fontsize=14, y=1.01,
    )

    for row, hemisphere in enumerate(HEMISPHERES):
        for col, (variant, color) in enumerate(zip(DIV1_VARIANTS, colors)):
            ax = axes[row, col]

            for label, ls, va_str in METHOD_INFO:
                x, y, tick_labels = get_b_annual_cycle(b_dataset, variant, hemisphere, va_str)
                ax.plot(x, y, linestyle=ls, color=color, marker='x',
                        label=label, markersize=5)

            # ── cosmetics ───────────────────────────────────────────────────
            ax.axhline(0, color='k', linewidth=0.6)
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(-0.2, 0.2)

            if row == 0:
                ax.set_title(variant, fontsize=10)

            if row == 1:
                ax.set_xticks(range(1, len(tick_labels) + 1))
                ax.set_xticklabels(tick_labels)

            if col == 0:
                hem_full = 'Northern Hemisphere' if hemisphere == 'n' else 'Southern Hemisphere'
                ax.set_ylabel(f'$\\bf{{{hem_full}}}$\nb', fontsize=10)

            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()

    fpath = os.path.join(plot_dir_b, 'b_va_method_comparison_NH_SH.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'Saved → {fpath}')
    plt.close(fig)


def plot_b_method_comparison_single_hem(b_dataset, hemisphere):
    """
    3-panel figure for a single hemisphere: columns = div1_QG / div1_QG_123 / div1_QG_gt3.
    Solid = no va, dashed = va, dotted = 500 hPa.
    """
    hem_label = HEM_LABELS[hemisphere]
    colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4),
        sharey=True,
        sharex=True,
    )
    fig.suptitle(
        f'b-parameter annual cycle (250–500–850 hPa) — va method comparison — {hem_label}',
        fontsize=13, y=1.02,
    )

    for col, (variant, color) in enumerate(zip(DIV1_VARIANTS, colors)):
        ax = axes[col]

        for label, ls, va_str in METHOD_INFO:
            x, y, tick_labels = get_b_annual_cycle(b_dataset, variant, hemisphere, va_str)
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

    fpath = os.path.join(plot_dir_b, f'b_va_method_comparison_{hem_label}.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'Saved → {fpath}')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
b_dataset = xr.open_dataset(b_path_250_850)

plot_b_method_comparison(b_dataset)
# for hem in HEMISPHERES:
#     plot_b_method_comparison_single_hem(b_dataset, hem)
