import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# ── paths ───────────────────────────────────────────────────────────────────

b_data = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/b_annual_cycle_cmip6_hist.csv'
efp_data = '/home/links/ct715/eddy_feedback/chapter1/cmip6/historical_runs/data/1979_2014/6h/efp_annual_cycle_cmip6_hist.csv'
PLOT_DIR = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/efp_vs_b/plots'

# ── load data ────────────────────────────────────────────────────────────────

df_b = pd.read_csv(b_data)
df_efp = pd.read_csv(efp_data)

# ── model overlap ────────────────────────────────────────────────────────────

models_b   = set(df_b['model'].unique())
models_efp = set(df_efp['model'].unique())
overlap    = sorted(models_b & models_efp)
only_b     = sorted(models_b - models_efp)
only_efp   = sorted(models_efp - models_b)

print(f"Models in b dataset:   {len(models_b)}")
print(f"Models in EFP dataset: {len(models_efp)}")
print(f"Overlapping models:    {len(overlap)}")
if only_b:
    print(f"Only in b:   {only_b}")
if only_efp:
    print(f"Only in EFP: {only_efp}")
print()

# ── merge on overlapping models ──────────────────────────────────────────────

df = pd.merge(df_b, df_efp, on=['model', 'variant', 'hemisphere', 'season'])
df = df[df['model'].isin(overlap)]

# ── constants ────────────────────────────────────────────────────────────────

VARIANTS = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
SEASONS  = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
            'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
HEMS     = {'n': 'NH', 's': 'SH'}

# assign a consistent colour per model
cmap        = plt.colormaps['tab20'].resampled(len(overlap))
model_color = {m: cmap(i) for i, m in enumerate(overlap)}

# ── helpers ──────────────────────────────────────────────────────────────────

def _add_scatter(ax, sub, title):
    """Scatter EFP vs b for each model; add regression line."""
    for model, grp in sub.groupby('model'):
        ax.scatter(grp['efp'], grp['b'], color=model_color[model],
                   s=25, alpha=0.85, label=model)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    # regression line + stats
    x, y = sub['efp'].values, sub['b'].values
    if len(x) > 1:
        m, c = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + c, color='k', linewidth=1.0)
        r, p = stats.pearsonr(x, y)
        p_str = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
        ax.text(0.05, 0.95, f'r={r:.2f}, {p_str}', transform=ax.transAxes,
                fontsize=6, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('EFP', fontsize=7)
    ax.set_ylabel('b', fontsize=7)
    ax.tick_params(labelsize=6)


def _make_legend(fig, ncol=4):
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=model_color[m], markersize=5, label=m)
               for m in overlap]
    fig.legend(handles=handles, loc='lower center', ncol=ncol,
               fontsize=6, frameon=True, bbox_to_anchor=(0.5, 0.0))


# ── 1. all-seasons plots (2 total: one per hemisphere) ──────────────────────

os.makedirs(os.path.join(PLOT_DIR, 'all_seasons'), exist_ok=True)

for hem_code, hem_name in HEMS.items():
    fig, axes = plt.subplots(12, 3, figsize=(13, 36))
    fig.suptitle(f'EFP vs b — {hem_name} — all seasons', fontsize=12, y=1.001)

    for row, season in enumerate(SEASONS):
        for col, variant in enumerate(VARIANTS):
            ax  = axes[row, col]
            sub = df[(df['hemisphere'] == hem_code) &
                     (df['season']     == season)   &
                     (df['variant']    == variant)]
            _add_scatter(ax, sub, f'{season} | {variant}')

    _make_legend(fig, ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(PLOT_DIR, 'all_seasons', f'efp_vs_b_{hem_code}_all_seasons.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

# ── 2. per-season plots (24 total: 12 seasons × 2 hemispheres) ──────────────

for hem_code, hem_name in HEMS.items():
    out_dir = os.path.join(PLOT_DIR, hem_code)
    os.makedirs(out_dir, exist_ok=True)

    for season in SEASONS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(f'EFP vs b — {hem_name} — {season}', fontsize=11)

        for col, variant in enumerate(VARIANTS):
            ax  = axes[col]
            sub = df[(df['hemisphere'] == hem_code) &
                     (df['season']     == season)   &
                     (df['variant']    == variant)]
            _add_scatter(ax, sub, variant)

        _make_legend(fig, ncol=5)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        path = os.path.join(out_dir, f'efp_vs_b_{hem_code}_{season}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")
