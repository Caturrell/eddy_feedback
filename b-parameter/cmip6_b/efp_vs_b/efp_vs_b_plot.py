import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# ── paths ───────────────────────────────────────────────────────────────────

b_data = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/b_annual_cycle_cmip6_hist.csv'
efp_data = '/home/links/ct715/eddy_feedback/chapter1/cmip6/historical_runs/data/1979_2014/6h/efp_annual_cycle_cmip6_hist.csv'
jra55_b_data = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/efp_vs_b/jra55_b_annual_cycle.csv'
jra55_efp_data = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/efp_vs_b/jra55_efp_annual_cycle.csv'
PLOT_DIR = '/home/links/ct715/eddy_feedback/b-parameter/cmip6_b/efp_vs_b/plots'

# ── load data ────────────────────────────────────────────────────────────────

df_b = pd.read_csv(b_data)
df_efp = pd.read_csv(efp_data)

df_jra55 = pd.merge(pd.read_csv(jra55_b_data), pd.read_csv(jra55_efp_data),
                     on=['model', 'variant', 'hemisphere', 'season'])
JRA55_COLOR = 'crimson'

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

def _pearson_stats(x, y):
    """Pearson r/p/n for two equal-length arrays (shared by scatter panels and heatmap)."""
    if len(x) > 1:
        r, p = stats.pearsonr(x, y)
        return r, p, len(x)
    return np.nan, np.nan, len(x)


def _add_scatter(ax, sub, title, hem_code=None, season=None, variant=None):
    """Scatter EFP vs b for each model; add regression line and JRA55 reference crosshair."""
    for model, grp in sub.groupby('model'):
        ax.scatter(grp['efp'], grp['b'], color=model_color[model],
                   s=25, alpha=0.85, label=model)

    if hem_code is not None:
        jra55_row = df_jra55[(df_jra55['hemisphere'] == hem_code) &
                              (df_jra55['season']     == season)   &
                              (df_jra55['variant']    == variant)]
        if len(jra55_row):
            ax.axhline(jra55_row['b'].values[0], color=JRA55_COLOR, linewidth=1.0, linestyle=':')
            ax.axvline(jra55_row['efp'].values[0], color=JRA55_COLOR, linewidth=1.0, linestyle=':')

    # regression line + stats
    x, y = sub['efp'].values, sub['b'].values
    if len(x) > 1:
        m, c = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + c, color='k', linewidth=1.0)
        r, p, n = _pearson_stats(x, y)
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
    handles.append(plt.Line2D([0], [0], color=JRA55_COLOR, linewidth=1.0,
                              linestyle=':', label='JRA55 (reanalysis)'))
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
            _add_scatter(ax, sub, f'{season} | {variant}',
                         hem_code=hem_code, season=season, variant=variant)

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
            _add_scatter(ax, sub, variant,
                         hem_code=hem_code, season=season, variant=variant)

        _make_legend(fig, ncol=5)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        path = os.path.join(out_dir, f'efp_vs_b_{hem_code}_{season}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

# ── 2b. main-season plot: NH DJF (top row) + SH JJA (bottom row) ────────────

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle('EFP vs b — Winter Seasons (NH: DJF; SH: JJA)', fontsize=12)

for row, (hem_code, hem_name, season) in enumerate([('n', 'NH', 'DJF'), ('s', 'SH', 'JJA')]):
    for col, variant in enumerate(VARIANTS):
        ax  = axes[row, col]
        sub = df[(df['hemisphere'] == hem_code) &
                 (df['season']     == season)   &
                 (df['variant']    == variant)]
        _add_scatter(ax, sub, f'{hem_name} {season} | {variant}',
                     hem_code=hem_code, season=season, variant=variant)

_make_legend(fig, ncol=5)
fig.tight_layout(rect=[0, 0.16, 1, 1])
path = os.path.join(PLOT_DIR, 'efp_vs_b_NH-DJF_SH-JJA.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {path}")

# ── 3. summary heatmap of all 72 season × hemisphere × variant correlations ─

VARIANT_LABELS = {'div1_QG': 'full', 'div1_QG_123': '1–3', 'div1_QG_gt3': '>3'}


def _build_corr_table():
    """Tidy table of Pearson r/p/n for every season × hemisphere × variant combo."""
    records = []
    for season in SEASONS:
        for hem_code, hem_name in HEMS.items():
            for variant in VARIANTS:
                sub = df[(df['hemisphere'] == hem_code) &
                         (df['season']     == season)   &
                         (df['variant']    == variant)]
                r, p, n = _pearson_stats(sub['efp'].values, sub['b'].values)
                records.append({
                    'season': season,
                    'hemisphere': hem_name,
                    'wavenumber_variant': variant,
                    'r': r,
                    'p': p,
                    'n': n,
                })
    return pd.DataFrame(records)


def _plot_corr_heatmap_single_hem(corr_df, hem_name, out_path):
    """Season × variant heatmap of Pearson r, for one fixed hemisphere."""
    ncols = len(VARIANTS)

    r_mat = np.full((len(SEASONS), ncols), np.nan)
    p_mat = np.full((len(SEASONS), ncols), np.nan)
    for i, season in enumerate(SEASONS):
        for j, variant in enumerate(VARIANTS):
            row = corr_df[(corr_df['season'] == season) &
                          (corr_df['hemisphere'] == hem_name) &
                          (corr_df['wavenumber_variant'] == variant)]
            if len(row):
                r_mat[i, j] = row['r'].values[0]
                p_mat[i, j] = row['p'].values[0]

    vmax = np.nanmax(np.abs(r_mat))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(4, 9.5))
    im = ax.imshow(r_mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(SEASONS) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=11)
    ax.set_yticks(np.arange(len(SEASONS)))
    ax.set_yticklabels(SEASONS, fontsize=11)

    for i in range(len(SEASONS)):
        for j in range(ncols):
            r_val, p_val = r_mat[i, j], p_mat[i, j]
            if np.isnan(r_val):
                continue
            bold = p_val < 0.05
            text_color = 'white' if abs(r_val) / vmax > 0.6 else 'black'
            if bold:
                label = rf'$\mathbf{{{r_val:.2f}^{{*}}}}$'
            else:
                label = f'{r_val:.2f}'
            fontsize = 8 if bold else 7
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='normal')

    cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.06)
    cbar.set_label('Pearson r', fontsize=9)

    fig.suptitle(f'EFP–b correlation (Pearson r)\n{hem_name} — by season, wavenumber filter',
                 fontsize=11, y=0.99)
    fig.text(0.5, 0.94, '* p<0.05 (bold)', fontsize=8, ha='center', va='top')

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_corr_heatmap(corr_df):
    """Condense all season/hemisphere/variant correlations into one heatmap."""
    col_hems     = [hem_name for hem_name in HEMS.values() for _ in VARIANTS]
    col_variants = [variant for _ in HEMS for variant in VARIANTS]
    ncols = len(col_hems)

    r_mat = np.full((len(SEASONS), ncols), np.nan)
    p_mat = np.full((len(SEASONS), ncols), np.nan)
    for i, season in enumerate(SEASONS):
        for j, (hem_name, variant) in enumerate(zip(col_hems, col_variants)):
            row = corr_df[(corr_df['season'] == season) &
                          (corr_df['hemisphere'] == hem_name) &
                          (corr_df['wavenumber_variant'] == variant)]
            if len(row):
                r_mat[i, j] = row['r'].values[0]
                p_mat[i, j] = row['p'].values[0]

    vmax = np.nanmax(np.abs(r_mat))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(8, 9.5))
    im = ax.imshow(r_mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    # white gridlines between cells
    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(SEASONS) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # divider between NH and SH blocks, extended slightly above the axes so it
    # also separates the NH/SH header labels like a table rule
    ax.axvline(len(VARIANTS) - 0.5, ymin=0, ymax=1.05, color='black', linewidth=3,
               zorder=5, clip_on=False)

    # tick labels
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in col_variants], fontsize=11)
    ax.set_yticks(np.arange(len(SEASONS)))
    ax.set_yticklabels(SEASONS, fontsize=11)

    # NH / SH block labels above each block
    ax.annotate('NH', xy=((len(VARIANTS) - 1) / 2, 1.02), xycoords=('data', 'axes fraction'),
                ha='center', va='bottom', fontsize=10, fontweight='bold', annotation_clip=False)
    ax.annotate('SH', xy=(len(VARIANTS) + (len(VARIANTS) - 1) / 2, 1.02), xycoords=('data', 'axes fraction'),
                ha='center', va='bottom', fontsize=10, fontweight='bold', annotation_clip=False)

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
                weight = 'normal'
            else:
                label = f'{r_val:.2f}'
                weight = 'normal'
            fontsize = 8 if bold else 7
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson r', fontsize=9)

    fig.suptitle('EFP–b correlation (Pearson r) by season, hemisphere, wavenumber filter',
                 fontsize=11, y=0.985)
    fig.text(0.5, 0.955, '* p<0.05 (bold)', fontsize=8, ha='center', va='top')

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = os.path.join(PLOT_DIR, 'all_seasons')
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, 'efp_vs_b_corr_heatmap.png')
    fig.savefig(png_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png_path}")

    csv_path = os.path.join(out_dir, 'efp_vs_b_corr_table.csv')
    corr_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


corr_df = _build_corr_table()
_plot_corr_heatmap(corr_df)

for hem_code, hem_name in HEMS.items():
    hem_out_dir = os.path.join(PLOT_DIR, 'all_seasons', hem_code)
    os.makedirs(hem_out_dir, exist_ok=True)

    png_path = os.path.join(hem_out_dir, 'efp_vs_b_corr_heatmap.png')
    _plot_corr_heatmap_single_hem(corr_df, hem_name, png_path)

    csv_path = os.path.join(hem_out_dir, 'efp_vs_b_corr_table.csv')
    corr_df[corr_df['hemisphere'] == hem_name].to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
