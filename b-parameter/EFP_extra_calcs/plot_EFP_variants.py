import json
import os
import string
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypalettes import load_cmap

# ── paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data', '1979_2014')
PLOTS    = os.path.join(HERE, 'plots')

LEVEL_VARIANTS = ('mean_level', '500hPa', '250_500_850hPa')
VARIANT_LABELS = {
    'mean_level':       'Mean level (200–600 hPa)',
    '500hPa':           '500 hPa',
    '250_500_850hPa':   '250/500/850 hPa mean',
}

# (panel title, NH config key, SH config key)
WAVENUMBER_PANELS = [
    ('All $k$',     'efp_nh',     'efp_sh'),
    ('$k = 1$–3', 'efp_nh_123', 'efp_sh_123'),
    ('$k > 3$',     'efp_nh_gt3', 'efp_sh_gt3'),
]

MONTH_MAP   = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J',
               7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}
MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
M2IDX       = {m: i for i, m in enumerate(MONTH_ORDER)}

PANEL_LABELS = list(string.ascii_lowercase)


def load_data(data_dir=DATA_DIR):
    data = {}
    for variant in LEVEL_VARIANTS:
        path = os.path.join(data_dir, f'efp_results_{variant}.json')
        with open(path) as f:
            data[variant] = json.load(f)
    return data


def extract_seasonal_series(config_dict):
    """Return (xs, ys) for the 12 rolling seasons, ordered Jul-Jun. Excludes ANN."""
    pairs = [(M2IDX[info['months'][1]], info['efp'])
             for season, info in config_dict.items() if season != 'ANN']
    pairs.sort()
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


ANNUAL_X = 13.5
DIVIDER_X = 12.5


def make_figure(data, config_key_by_panel, hemi_label, save_path):
    cmap = load_cmap('highcontrast')
    colors = {variant: cmap.colors[i] for i, variant in enumerate(LEVEL_VARIANTS)}

    n_panels = len(config_key_by_panel)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.2), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for i, (title, config_key) in enumerate(config_key_by_panel):
        ax = axes[i]

        for variant in LEVEL_VARIANTS:
            config_dict = data[variant][config_key]
            xs, ys = extract_seasonal_series(config_dict)
            ax.plot(xs, ys, marker='o', markersize=5, linewidth=2,
                    color=colors[variant])

            ann_value = config_dict['ANN']['efp']
            ax.scatter([ANNUAL_X], [ann_value], marker='D', s=55,
                       color=colors[variant], zorder=3, edgecolor='k', linewidth=0.4)

        ax.axvline(DIVIDER_X, color='0.3', linewidth=2.5, zorder=2)

        ax.set_xticks(list(range(12)) + [ANNUAL_X])
        ax.set_xticklabels([MONTH_MAP[m] for m in MONTH_ORDER] + ['Annual'])
        ax.set_xlim(-0.6, ANNUAL_X + 0.6)
        ax.grid(True, axis='y', alpha=0.4)
        ax.tick_params(labelsize=10)

        ax.set_title(f'$\\bf{{({PANEL_LABELS[i]})}}$ {title}', fontsize=13)
        if i == 0:
            ax.set_ylabel('EFP', fontsize=13)
        ax.set_xlabel('Month', fontsize=12)

    legend_handles = [
        Line2D([0], [0], color=colors[v], lw=2, marker='o', label=VARIANT_LABELS[v])
        for v in LEVEL_VARIANTS
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(LEVEL_VARIANTS),
               bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=11)

    fig.suptitle(f'EFP level-variant comparison — {hemi_label} (1979–2014)',
                 fontsize=14, y=1.03)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.16)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    data = load_data()

    make_figure(
        data,
        [(title, nh_key) for title, nh_key, sh_key in WAVENUMBER_PANELS],
        hemi_label='Northern Hemisphere',
        save_path=os.path.join(PLOTS, 'efp_variants_nh.png'),
    )
    make_figure(
        data,
        [(title, sh_key) for title, nh_key, sh_key in WAVENUMBER_PANELS],
        hemi_label='Southern Hemisphere',
        save_path=os.path.join(PLOTS, 'efp_variants_sh.png'),
    )


if __name__ == "__main__":
    main()
