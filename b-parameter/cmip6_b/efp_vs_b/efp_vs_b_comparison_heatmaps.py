"""
Cross-wavenumber comparison: for each b wavenumber variant (full, 1-3, >3),
correlate it against ALL THREE EFP wavenumber variants (not just the matching
one), for every season × hemisphere. Produces one summary heatmap per b
variant. Scatter plots are intentionally not produced here -- see
efp_vs_b_plot.py for the matched-variant scatter plots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# ── paths ───────────────────────────────────────────────────────────────────

b_data = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/b_annual_cycle_cmip6_hist.csv'
efp_data = '/home/links/ct715/eddy_feedback/chapter1/cmip6/historical_runs/data/1979_2014/6h/efp_annual_cycle_cmip6_hist.csv'
OUT_DIR = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/efp_vs_b/plots/comparison_heatmaps'

os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────

df_b = pd.read_csv(b_data)
df_efp = pd.read_csv(efp_data)

# ── model overlap ────────────────────────────────────────────────────────────

models_b   = set(df_b['model'].unique())
models_efp = set(df_efp['model'].unique())
overlap    = sorted(models_b & models_efp)

print(f"Models in b dataset:   {len(models_b)}")
print(f"Models in EFP dataset: {len(models_efp)}")
print(f"Overlapping models:    {len(overlap)}")
print()

# ── cross-join b-variant × efp-variant on (model, hemisphere, season) ───────
# Merging without 'variant' as a join key gives every (variant_b, variant_efp)
# combination per model/hemisphere/season -- exactly the 3x3 cross comparison
# ("b full vs EFP full/1-3/>3", "b 1-3 vs EFP full/1-3/>3", etc.).

df = pd.merge(df_b, df_efp, on=['model', 'hemisphere', 'season'], suffixes=('_b', '_efp'))
df = df[df['model'].isin(overlap)]

# ── constants ────────────────────────────────────────────────────────────────

VARIANTS       = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
VARIANT_LABELS = {'div1_QG': 'full', 'div1_QG_123': '1–3', 'div1_QG_gt3': '>3'}
VARIANT_FILE   = {'div1_QG': 'full', 'div1_QG_123': '123', 'div1_QG_gt3': 'gt3'}
SEASONS  = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
            'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
HEMS     = {'n': 'NH', 's': 'SH'}

# ── helpers ──────────────────────────────────────────────────────────────────

def _pearson_stats(x, y):
    """Pearson r/p/n for two equal-length arrays."""
    if len(x) > 1:
        r, p = stats.pearsonr(x, y)
        return r, p, len(x)
    return np.nan, np.nan, len(x)


def _build_corr_table(b_variant, hem_code):
    """Tidy table of Pearson r/p/n between b_variant and every efp_variant × season combo, for one hemisphere."""
    records = []
    for season in SEASONS:
        for efp_variant in VARIANTS:
            sub = df[(df['hemisphere']  == hem_code)   &
                     (df['season']      == season)     &
                     (df['variant_b']   == b_variant)  &
                     (df['variant_efp'] == efp_variant)]
            r, p, n = _pearson_stats(sub['efp'].values, sub['b'].values)
            records.append({
                'season': season,
                'efp_variant': efp_variant,
                'r': r,
                'p': p,
                'n': n,
            })
    return pd.DataFrame(records)


def _plot_corr_heatmap(corr_df, b_variant, hem_name, out_path):
    """Season × efp_variant heatmap of Pearson r, for one fixed b_variant and hemisphere."""
    ncols = len(VARIANTS)

    r_mat = np.full((len(SEASONS), ncols), np.nan)
    p_mat = np.full((len(SEASONS), ncols), np.nan)
    for i, season in enumerate(SEASONS):
        for j, variant in enumerate(VARIANTS):
            row = corr_df[(corr_df['season'] == season) &
                          (corr_df['efp_variant'] == variant)]
            if len(row):
                r_mat[i, j] = row['r'].values[0]
                p_mat[i, j] = row['p'].values[0]

    vmax = np.nanmax(np.abs(r_mat))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(4, 9.5))
    im = ax.imshow(r_mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    # white gridlines between cells
    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(SEASONS) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # tick labels
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=11)
    ax.set_yticks(np.arange(len(SEASONS)))
    ax.set_yticklabels(SEASONS, fontsize=11)

    # annotate each cell with r (+ significance marker)
    for i in range(len(SEASONS)):
        for j in range(ncols):
            r_val, p_val = r_mat[i, j], p_mat[i, j]
            if np.isnan(r_val):
                continue
            bold = p_val < 0.05
            text_color = 'white' if abs(r_val) / vmax > 0.6 else 'black'
            if bold:
                # mathtext ignores fontweight, so bold must be applied via \mathbf
                label = rf'$\mathbf{{{r_val:.2f}^{{*}}}}$'
            else:
                label = f'{r_val:.2f}'
            fontsize = 8 if bold else 7
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='normal')

    cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.06)
    cbar.set_label('Pearson r', fontsize=9)

    fig.suptitle(f'b ({VARIANT_LABELS[b_variant]}) vs EFP correlation (Pearson r)\n'
                 f'{hem_name} — by season, EFP wavenumber filter', fontsize=11, y=0.99)
    fig.text(0.5, 0.94, '* p<0.05 (bold)', fontsize=8, ha='center', va='top')

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── one heatmap per hemisphere × b-variant, comparing against all 3 EFP variants ─

for hem_code, hem_name in HEMS.items():
    hem_out_dir = os.path.join(OUT_DIR, hem_code)
    os.makedirs(hem_out_dir, exist_ok=True)

    for b_variant in VARIANTS:
        corr_df = _build_corr_table(b_variant, hem_code)

        png_path = os.path.join(hem_out_dir, f'b_{VARIANT_FILE[b_variant]}_vs_efp_heatmap.png')
        _plot_corr_heatmap(corr_df, b_variant, hem_name, png_path)

        csv_path = os.path.join(hem_out_dir, f'b_{VARIANT_FILE[b_variant]}_vs_efp_corr_table.csv')
        corr_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
