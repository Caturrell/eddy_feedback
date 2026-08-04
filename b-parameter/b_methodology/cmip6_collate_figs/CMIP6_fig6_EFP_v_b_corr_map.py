"""
Correlation heatmap: b-parameter (250/500/850hPa, "va" method) vs EFP
(500hPa), matched by wavenumber variant (full, 1-3, >3), across the CMIP6
historical models common to both datasets. One heatmap per hemisphere (NH,
SH), season along the x-axis, variant along the y-axis.

Also builds a Southern-Hemisphere-only composite figure that stacks two rows
of violin plots (CMIP6 model spread of EFP, then b, per season x wavenumber
band, with the JRA-55 reanalysis value marked on each violin) above the SH
heatmap panel.

b source data: b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/
               <model>/6hrPlevPt/b_dataset.nc
               (same b calculation used in CMIP6_fig5_tau_bar.py --
               Simpson et al. 2013 lag-regression "va" method, vertically
               averaged over 250/500/850hPa; all 12 seasons and both
               hemispheres are extracted here, vs. fig5's all_time/JJA/DJF/
               NDJ/southern-hemisphere-only subset)
EFP source data: chapter1/cmip6/historical_runs/data/1979_2014/6h/
                 efp_annual_cycle_cmip6_hist.csv
                 (500hPa-only EFP, 1979-2014, 6-hourly, per-model)
JRA-55 reference data (composite figure violin markers only):
    b-parameter/cmip6_b/efp_vs_b/jra55_b_annual_cycle.csv
    b-parameter/cmip6_b/efp_vs_b/jra55_efp_annual_cycle.csv
"""
import os
import numpy as np
import pandas as pd
import xarray as xar
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))

b_base_dir = os.path.normpath(os.path.join(
    script_dir, '..', '..', 'cmip6_b', '250-500-850hPa_dm', '1979_2015'
))
efp_csv = os.path.normpath(os.path.join(
    script_dir, '..', '..', '..', 'chapter1', 'cmip6', 'historical_runs',
    'data', '1979_2014', '6h', 'efp_annual_cycle_cmip6_hist.csv'
))
jra55_b_csv = os.path.normpath(os.path.join(
    script_dir, '..', '..', 'cmip6_b', 'efp_vs_b', 'jra55_b_annual_cycle.csv'
))
jra55_efp_csv = os.path.normpath(os.path.join(
    script_dir, '..', '..', 'cmip6_b', 'efp_vs_b', 'jra55_efp_annual_cycle.csv'
))
OUT_DIR = os.path.join(script_dir, 'plots', 'fig6_efp_vs_b_corr_map')
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS       = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
VARIANT_LABELS = {'div1_QG': r'all $k$', 'div1_QG_123': r'$k1$-3', 'div1_QG_gt3': r'$k>3$'}
# Matches the wavenumber-band colours used in fig5_m-regression_b-parameter.py.
VARIANT_COLORS = {'div1_QG': 'tab:blue', 'div1_QG_123': 'tab:orange', 'div1_QG_gt3': 'tab:green'}
SEASONS = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
           'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
HEMS = {'n': 'Northern Hemisphere', 's': 'Southern Hemisphere'}

# ── load b data directly from each model's b_dataset.nc ─────────────────────

model_names = sorted(
    d for d in os.listdir(b_base_dir)
    if os.path.isdir(os.path.join(b_base_dir, d))
)

b_records = []
skipped_models = []
for model in model_names:
    b_file = os.path.join(b_base_dir, model, '6hrPlevPt', 'b_dataset.nc')
    if not os.path.isfile(b_file):
        skipped_models.append(model)
        continue
    with xar.open_dataset(b_file) as b_ds:
        for variant in VARIANTS:
            for hem_code in HEMS:
                for season in SEASONS:
                    var_name = f'ucomp_va_{variant}_va_b_{hem_code}_{season}'
                    if var_name not in b_ds:
                        continue
                    b_records.append({
                        'model': model,
                        'variant': variant,
                        'hemisphere': hem_code,
                        'season': season,
                        'b': float(b_ds[var_name].mean('lag', skipna=True)),
                    })
if skipped_models:
    print(f"Warning: skipped {len(skipped_models)} model(s) with no b_dataset.nc: "
          f"{', '.join(skipped_models)}")

df_b = pd.DataFrame(b_records)
print(f"b data: {df_b['model'].nunique()} models")

# ── load EFP data ────────────────────────────────────────────────────────────

df_efp = pd.read_csv(efp_csv)
print(f"EFP data: {df_efp['model'].nunique()} models")

# ── model overlap ────────────────────────────────────────────────────────────

models_b   = set(df_b['model'].unique())
models_efp = set(df_efp['model'].unique())
overlap    = sorted(models_b & models_efp)

print(f"Overlapping models: {len(overlap)}")
print()

df_b   = df_b[df_b['model'].isin(overlap)]
df_efp = df_efp[df_efp['model'].isin(overlap)]

# ── merge on matched variant (b full <-> EFP full, etc. -- not a cross join) ─

df = pd.merge(df_b, df_efp, on=['model', 'variant', 'hemisphere', 'season'],
              suffixes=('_b', '_efp'))

# ── JRA-55 reference data (violin markers in the composite figure only) ─────

df_jra55_b   = pd.read_csv(jra55_b_csv)
df_jra55_efp = pd.read_csv(jra55_efp_csv)

# ── helpers: correlation table + heatmap panel ───────────────────────────────

def _pearson_stats(x, y):
    """Pearson r/p/n for two equal-length arrays."""
    if len(x) > 1:
        r, p = stats.pearsonr(x, y)
        return r, p, len(x)
    return np.nan, np.nan, len(x)


def _build_corr_table(hem_code):
    """Tidy table of Pearson r/p/n between matched b/EFP variants, for every season, one hemisphere."""
    records = []
    for variant in VARIANTS:
        for season in SEASONS:
            sub = df[(df['hemisphere'] == hem_code) &
                     (df['season']     == season)  &
                     (df['variant']    == variant)]
            r, p, n = _pearson_stats(sub['efp'].values, sub['b'].values)
            records.append({'variant': variant, 'season': season, 'r': r, 'p': p, 'n': n})
    return pd.DataFrame(records)


def _draw_corr_heatmap(ax, corr_df, hem_name, panel_label=True):
    """Variant x season heatmap of Pearson r (matched-variant b vs EFP), one hemisphere, onto a given ax."""
    nrows = len(VARIANTS)
    ncols = len(SEASONS)

    r_mat = np.full((nrows, ncols), np.nan)
    p_mat = np.full((nrows, ncols), np.nan)
    for i, variant in enumerate(VARIANTS):
        for j, season in enumerate(SEASONS):
            row = corr_df[(corr_df['variant'] == variant) & (corr_df['season'] == season)]
            if len(row):
                r_mat[i, j] = row['r'].values[0]
                p_mat[i, j] = row['p'].values[0]

    vmax = np.nanmax(np.abs(r_mat))
    vmin = -vmax

    im = ax.imshow(r_mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    # white gridlines between cells
    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # tick labels
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(SEASONS, fontsize=11)
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=11)

    # annotate each cell with r (+ significance marker)
    for i in range(nrows):
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
            fontsize = 12 if bold else 11
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='normal')

    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal', location='bottom',
                               fraction=0.06, pad=0.22)
    cbar.set_label('Pearson r', fontsize=9)

    title = r'$b$ vs. EFP correlation, * p<0.05 (bold) - ' + hem_name
    if panel_label:
        title = r'$\mathbf{(c)}$ ' + title
    ax.set_title(title, fontsize=12)


def _plot_corr_heatmap(corr_df, hem_name, out_path, panel_label=True):
    """Standalone single-panel heatmap figure for one hemisphere."""
    fig, ax = plt.subplots(figsize=(0.85 * len(SEASONS) + 1.5, 0.85 * len(VARIANTS) + 2.))
    _draw_corr_heatmap(ax, corr_df, hem_name, panel_label=panel_label)

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── helpers: violin panels (CMIP6 model spread + JRA-55 marker) ─────────────

VARIANT_OFFSETS = {'div1_QG': -0.25, 'div1_QG_123': 0.0, 'div1_QG_gt3': 0.25}
VIOLIN_WIDTH = 0.2


def _violin_data(hem_code, value_col):
    """dict[(variant, season)] -> np.array of per-model values, for one hemisphere."""
    sub_all = df[df['hemisphere'] == hem_code]
    data = {}
    for variant in VARIANTS:
        for season in SEASONS:
            sub = sub_all[(sub_all['variant'] == variant) & (sub_all['season'] == season)]
            data[(variant, season)] = sub[value_col].dropna().values
    return data


def _jra55_lookup(df_jra55, hem_code, value_col):
    """dict[(variant, season)] -> single JRA-55 reference value, for one hemisphere."""
    sub = df_jra55[df_jra55['hemisphere'] == hem_code]
    return {(row.variant, row.season): getattr(row, value_col) for row in sub.itertuples()}


def _draw_violin_row(ax, hem_code, value_col, df_jra55, ylabel, zero_line=False, ylim=None):
    """Season x wavenumber-band violin row: CMIP6 model spread, JRA-55 marked, onto a given ax."""
    data = _violin_data(hem_code, value_col)
    jra55_vals = _jra55_lookup(df_jra55, hem_code, value_col)

    for variant in VARIANTS:
        offset = VARIANT_OFFSETS[variant]
        color = VARIANT_COLORS[variant]

        positions, datasets = [], []
        for j, season in enumerate(SEASONS):
            vals = data[(variant, season)]
            if len(vals) > 1:
                positions.append(j + offset)
                datasets.append(vals)

        if datasets:
            parts = ax.violinplot(datasets, positions=positions, widths=VIOLIN_WIDTH,
                                   showmeans=False, showmedians=True, showextrema=True)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.55)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                parts[key].set_color(color)
                parts[key].set_linewidth(1.)

        jra55_x = [j + offset for j in range(len(SEASONS))]
        jra55_y = [jra55_vals.get((variant, season), np.nan) for season in SEASONS]
        ax.scatter(jra55_x, jra55_y, marker='*', s=150, color='black',
                   edgecolor='white', linewidth=0.5, zorder=5)

    if zero_line:
        ax.axhline(0, color='k', linewidth=0.5, zorder=1)

    ax.set_xlim(-0.5, len(SEASONS) - 0.5)
    ax.set_xticks(np.arange(len(SEASONS)))
    ax.set_xticklabels(SEASONS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, axis='y', alpha=0.4)
    ax.tick_params(labelsize=9)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _draw_efp_violins(ax, hem_code, hem_name):
    _draw_violin_row(ax, hem_code, 'efp', df_jra55_efp, ylabel='EFP', zero_line=False)
    ax.set_title(rf'$\mathbf{{(a)}}$ Eddy feedback parameter - CMIP6 model spread - {hem_name}',
                 fontsize=13, pad=12)


def _draw_b_violins(ax, hem_code, hem_name, ylim=None):
    _draw_violin_row(ax, hem_code, 'b', df_jra55_b, ylabel=r'$b$', zero_line=True, ylim=ylim)
    ax.set_title(rf'$\mathbf{{(b)}}$ $b$-parameter - CMIP6 model spread - {hem_name}',
                 fontsize=13, pad=12)


# ── composite figure: EFP violins / b violins / heatmap, one hemisphere ─────

def build_composite_figure(hem_code, hem_name, out_path, b_ylim=None):
    corr_df = _build_corr_table(hem_code)

    fig = plt.figure(figsize=(0.85 * len(SEASONS) + 1.5, 13.))
    # Two independent grids so the tight EFP/legend/b spacing doesn't also
    # squeeze the gap above the heatmap (which needs room for its own title
    # and, now, a horizontal colorbar below it).
    # Row 1 = spacer reserved for the shared legend, sitting between the two violin rows.
    gs_violins = fig.add_gridspec(3, 1, height_ratios=[1, 0.28, 1], hspace=0.12,
                                   top=0.97, bottom=0.40, left=0.07, right=0.97)
    gs_heat = fig.add_gridspec(1, 1, top=0.32, bottom=0.04, left=0.07, right=0.97)

    ax_efp  = fig.add_subplot(gs_violins[0, 0])
    ax_b    = fig.add_subplot(gs_violins[2, 0], sharex=ax_efp)
    ax_heat = fig.add_subplot(gs_heat[0, 0], sharex=ax_efp)

    _draw_efp_violins(ax_efp, hem_code, hem_name)
    _draw_b_violins(ax_b, hem_code, hem_name, ylim=b_ylim)
    _draw_corr_heatmap(ax_heat, corr_df, hem_name)

    variant_handles = [Patch(facecolor=VARIANT_COLORS[v], edgecolor=VARIANT_COLORS[v],
                              alpha=0.55, label=VARIANT_LABELS[v]) for v in VARIANTS]
    jra55_handle = Line2D([0], [0], marker='*', color='none', markerfacecolor='black',
                          markeredgecolor='white', markersize=10, label='JRA-55')
    legend_bbox = gs_violins[1, 0].get_position(fig)
    legend_y = (legend_bbox.y0 + legend_bbox.y1) / 2
    fig.legend(handles=variant_handles + [jra55_handle], loc='center',
               bbox_to_anchor=(0.5, legend_y), ncol=4, fontsize=9, frameon=False)

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_violin_pair_figure(hem_code, hem_name, out_path):
    """Panels (a) EFP and (b) b only -- same drawing logic/styling as build_composite_figure,
    just without the (c) heatmap panel."""
    fig = plt.figure(figsize=(0.85 * len(SEASONS) + 1.5, 9.))
    gs_violins = fig.add_gridspec(3, 1, height_ratios=[1, 0.28, 1], hspace=0.12,
                                   top=0.95, bottom=0.06, left=0.07, right=0.97)

    ax_efp = fig.add_subplot(gs_violins[0, 0])
    ax_b   = fig.add_subplot(gs_violins[2, 0], sharex=ax_efp)

    _draw_efp_violins(ax_efp, hem_code, hem_name)
    _draw_b_violins(ax_b, hem_code, hem_name)

    variant_handles = [Patch(facecolor=VARIANT_COLORS[v], edgecolor=VARIANT_COLORS[v],
                              alpha=0.55, label=VARIANT_LABELS[v]) for v in VARIANTS]
    jra55_handle = Line2D([0], [0], marker='*', color='none', markerfacecolor='black',
                          markeredgecolor='white', markersize=10, label='JRA-55')
    legend_bbox = gs_violins[1, 0].get_position(fig)
    legend_y = (legend_bbox.y0 + legend_bbox.y1) / 2
    fig.legend(handles=variant_handles + [jra55_handle], loc='center',
               bbox_to_anchor=(0.5, legend_y), ncol=4, fontsize=9, frameon=False)

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── one standalone heatmap per hemisphere (unchanged) ────────────────────────

for hem_code, hem_name in HEMS.items():
    corr_df = _build_corr_table(hem_code)

    png_path = os.path.join(OUT_DIR, f'efp_vs_b_corr_heatmap_{hem_code}.png')
    _plot_corr_heatmap(corr_df, hem_name, png_path)

    csv_path = os.path.join(OUT_DIR, f'efp_vs_b_corr_table_{hem_code}.csv')
    corr_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

# ── composite figures (violins + heatmap), one per hemisphere, into OUT_DIR ─

build_composite_figure('s', HEMS['s'], os.path.join(OUT_DIR, 'efp_vs_b_corr_composite_s.png'))
build_composite_figure('n', HEMS['n'], os.path.join(OUT_DIR, 'efp_vs_b_corr_composite_n.png'),
                        b_ylim=(-0.4, 0.4))

# ── SH main plot, split into (a+b) violins and (c) heatmap, top-level ───────

corr_df_s = _build_corr_table('s')
build_violin_pair_figure('s', HEMS['s'],
                          os.path.join(script_dir, 'plots', 'CMIP6_fig6a_efp-b_corr_map_spread.png'))
_plot_corr_heatmap(corr_df_s, HEMS['s'],
                    os.path.join(script_dir, 'plots', 'CMIP6_fig6b_efp-b_corr_map_spread.png'),
                    panel_label=False)
