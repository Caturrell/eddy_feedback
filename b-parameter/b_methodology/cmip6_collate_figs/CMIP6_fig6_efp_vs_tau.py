"""
Companion to CMIP6_fig5_tau_bar.py: same design (per-season tau bar charts,
EFP-vs-tau scatter panels for three spatial/wavenumber scales, JRA55
reference, cumulative outlier-exclusion sets, JJA/DJF comparison figures,
Pearson r/p summary CSV) but plots the eddy feedback parameter (EFP) against
tau instead of the b-parameter.

Unlike tau and b, EFP has no all-time value anywhere in the repo for either
CMIP6 or JRA55 -- only the 12 rolling 3-month seasons exist (the underlying
efp_results_500hPa.json has no 'all_time' key). So, unlike CMIP6_fig5_tau_bar.py,
this script covers JJA and DJF only; there is no all_time/ section and no
three-way seasonal comparison, just a JJA-vs-DJF one.

JRA55 EFP uses jra55_efp_annual_cycle.csv, which was built specifically to
match the CMIP6 EFP methodology: div1_QG (QG-filtered divergence), 500hPa
only, 6-hourly, 1979-2014 -- the same choices reflected in
efp_annual_cycle_cmip6_hist.csv (use_500hPa_only=True in hist_calc_efp_b.py).
Other JRA55 EFP variants exist (daily vs 6-hourly, div1_pr vs div1_QG,
1958-2016/1979-2016/1979-2014, full-column vs 500hPa -- see
chapter1/reanalysis/data/jra55_efp_ALL_variations.csv) but are intentionally
not used here since they wouldn't be methodologically comparable to the
CMIP6 side.

Models are sorted alphabetically (matching the model legend in
CMIP6_fig1_cospec_coher_pdiff.py and CMIP6_fig5_tau_bar.py) and use the same
turbo colormap colours.

Cumulative outlier-exclusion sets (see CMIP6_fig5_tau_bar.py for why):
  - remove_outlier: EC-Earth3-CC excluded (27 models)
  - remove_fgoals: EC-Earth3-CC AND FGOALS-f3-L excluded (26 models)

Output layout under plots/fig6_efp_vs_tau/:
  - JJA/ and DJF/ (each with remove_outlier/ and remove_fgoals/ subfolders):
    a single-row tau bar chart (with JRA55 reference line) and EFP-vs-tau
    scatter panels (main / stacked) for that season.
  - directly in fig6_efp_vs_tau/ (and its own remove_outlier/, remove_fgoals/):
    CMIP6_fig6_tau_bar_jja_djf.png (single-panel grouped bar chart, two bars
    per model/mean, one colour per season, JRA55 drawn as two matching
    reference lines) and CMIP6_fig6_efp_vs_tau_jja_djf.png (two scatter rows,
    JJA above DJF, each sharing its row's tau axis, panels labelled (a)-(f)).
  - CMIP6_fig6_efp_vs_tau_correlations.csv: Pearson r/p for every
    (season x variant x model-exclusion set) combination.

Tau source data: b-parameter/b_methodology/all_plots_true/250-500-850hPa_dm/
                 1979_2015/<model>/6hrPlevPt/power_spec.nc
JRA55 tau reference data: b-parameter/b_methodology/tau_values/data/jra55_tau_fit_3.csv
EFP source data: chapter1/cmip6/historical_runs/data/1979_2014/6h/efp_annual_cycle_cmip6_hist.csv
JRA55 EFP reference data: b-parameter/cmip6_b/efp_vs_b/jra55_efp_annual_cycle.csv
"""

import os
import csv
import numpy as np
import xarray as xar
import matplotlib.pyplot as plt
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))

cmip6_base_dir = os.path.normpath(os.path.join(
    script_dir, '..', 'all_plots_true', '250-500-850hPa_dm', '1979_2015'
))

hemisphere = 's'

efp_variants = {
    'full': 'div1_QG',
    'k1-3': 'div1_QG_123',
    'gt3': 'div1_QG_gt3',
}

variant_titles = {
    'full': 'All Wavenumbers',
    'k1-3': 'Wavenumbers 1-3',
    'gt3': 'Wavenumbers >3',
}

SEASONS = ['JJA', 'DJF']

# Cumulative: each set excludes everything the previous set(s) excluded, plus
# its own model(s) -- 'remove_outlier' drops EC-Earth3-CC (27 models),
# 'remove_fgoals' drops EC-Earth3-CC AND FGOALS-f3-L on top of that (26).
OUTLIER_SETS = [
    ('remove_outlier', ['EC-Earth3-CC']),
    ('remove_fgoals', ['EC-Earth3-CC', 'FGOALS-f3-L']),
]


def _stacked_suffix(tag):
    return '' if tag == 'remove_outlier' else f'_{tag.split("_", 1)[1]}'


def _excluded_suffix(outlier_models):
    return f' ({", ".join(outlier_models)} excluded)'


model_names = sorted(
    d for d in os.listdir(cmip6_base_dir)
    if os.path.isdir(os.path.join(cmip6_base_dir, d))
)


def _load_tau_data(time_frame):
    ucomp_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
    div1_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'
    tau_var = f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'

    data = {}
    skipped = []
    for model in model_names:
        power_spec_file = os.path.join(cmip6_base_dir, model, '6hrPlevPt', 'power_spec.nc')
        if not os.path.isfile(power_spec_file):
            skipped.append(model)
            continue
        with xar.open_dataset(power_spec_file, auto_complex=True) as power_spec_ds:
            try:
                data[model] = float(power_spec_ds[tau_var])
            except KeyError:
                skipped.append(model)
    if skipped:
        print(f'Warning: skipped {len(skipped)} model(s) with no usable power_spec.nc '
              f'for tau ({time_frame}): {", ".join(skipped)}')
    return data


def _load_jra55_tau(csv_path, season, wind_variant='va'):
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if (row['hemisphere'] == hemisphere and row['season'] == season
                    and row['wind_variant'] == wind_variant and row['qg_mode'] == 'div1_QG'):
                return float(row['tau_fit_3'])
    raise ValueError(
        f'No JRA55 tau found for season={season}, hemisphere={hemisphere}, '
        f'wind_variant={wind_variant} in {csv_path}'
    )


def _load_efp_data(csv_path, season, models):
    data = {key: {} for key in efp_variants}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if (row['hemisphere'] != hemisphere or row['season'] != season
                    or row['model'] not in models):
                continue
            for key, variant in efp_variants.items():
                if row['variant'] == variant:
                    data[key][row['model']] = float(row['efp'])
    return data


def _load_jra55_efp(csv_path, season):
    values = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row['hemisphere'] != hemisphere or row['season'] != season:
                continue
            for key, variant in efp_variants.items():
                if row['variant'] == variant:
                    values[key] = float(row['efp'])
    return values


# ---------------------------------------------------------------------------
# Base data: tau (per season, all models) so we know which models to use.
# ---------------------------------------------------------------------------
tau_maps = {season: _load_tau_data(season) for season in SEASONS}
used_models = sorted(set.intersection(*[set(tau_maps[s]) for s in SEASONS]))
print(f'Plotting {len(used_models)} model(s): {", ".join(used_models)}')

jra55_tau_file = os.path.normpath(
    os.path.join(script_dir, '..', 'tau_values', 'data', 'jra55_tau_fit_3.csv')
)
jra55_tau_maps = {season: _load_jra55_tau(jra55_tau_file, season) for season in SEASONS}
print(f'Loaded JRA55 tau: {jra55_tau_maps}')

n_models = len(used_models)
model_colors = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, n_models))
model_color_map = dict(zip(used_models, model_colors))

efp_file = os.path.normpath(os.path.join(
    script_dir, '..', '..', '..', 'chapter1', 'cmip6', 'historical_runs', 'data',
    '1979_2014', '6h', 'efp_annual_cycle_cmip6_hist.csv'
))
jra55_efp_file = os.path.normpath(
    os.path.join(script_dir, '..', '..', 'cmip6_b', 'efp_vs_b', 'jra55_efp_annual_cycle.csv')
)

efp_maps = {season: _load_efp_data(efp_file, season, used_models) for season in SEASONS}
jra55_efp_maps = {season: _load_jra55_efp(jra55_efp_file, season) for season in SEASONS}
print(f'Loaded JRA55 EFP: {jra55_efp_maps}')

scatter_models = {
    season: [
        model for model in used_models
        if model in tau_maps[season] and all(model in efp_maps[season][key] for key in efp_variants)
    ]
    for season in SEASONS
}
for season in SEASONS:
    print(f'Scattering {season} EFP vs tau for {len(scatter_models[season])} model(s): '
          f'{", ".join(scatter_models[season])}')

jra55_points = {
    season: {key: (jra55_efp_maps[season][key], jra55_tau_maps[season]) for key in efp_variants}
    for season in SEASONS
}

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
for season in SEASONS:
    season_tag = season.lower()
    efp_csv_file = os.path.join(data_dir, f'CMIP6_efp_{season_tag}.csv')
    with open(efp_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'efp_full', 'efp_k1-3', 'efp_gt3'])
        for model in scatter_models[season]:
            writer.writerow([
                model,
                f'{efp_maps[season]["full"][model]:.4f}',
                f'{efp_maps[season]["k1-3"][model]:.4f}',
                f'{efp_maps[season]["gt3"][model]:.4f}',
            ])
    print(f'Saved {efp_csv_file}')

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)
subplot_dir = os.path.join(plot_dir, 'fig6_efp_vs_tau')
os.makedirs(subplot_dir, exist_ok=True)
outlier_dirs = {}
for _tag, _ in OUTLIER_SETS:
    _d = os.path.join(subplot_dir, _tag)
    os.makedirs(_d, exist_ok=True)
    outlier_dirs[_tag] = _d


# ---------------------------------------------------------------------------
# Vertical bar chart of tau, with a black multi-model-mean bar, one per
# season (no all-time row/panel -- see module docstring).
# ---------------------------------------------------------------------------
def _draw_tau_bar(ax, models, values_map, title, jra55_value=None):
    n = len(models)
    values = [values_map[model] for model in models]
    mmm = float(np.mean(values))

    x_pos = np.arange(n + 1)
    bar_colors = [model_color_map[model] for model in models] + ['k']
    bar_values = values + [mmm]
    bar_labels = models + ['Multi-model\nmean']

    bars = ax.bar(x_pos, bar_values, color=bar_colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, rotation=90, fontsize=8)

    for bar, value in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                 f'{value:4.2f}', ha='center', va='bottom', fontsize=7,
                 rotation=90)

    handles = []
    if jra55_value is not None:
        h_jra55 = ax.axhline(
            jra55_value, color='k', lw=2., linestyle='-',
            label=rf'JRA55 ($\tau$={jra55_value:4.2f}d)'
        )
        handles.append(h_jra55)

    ax.set_ylabel(r'$\tau$ (days)', fontsize=11)
    ax.set_ylim(0., max(bar_values + ([jra55_value] if jra55_value is not None else [])) + 2.)
    ax.grid(True, axis='y')
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=True)
    ax.set_title(title, fontsize=13)


def _plot_tau_bar(models, tau_map, jra55_value, season, out_file, title_suffix=''):
    fig, ax = plt.subplots(figsize=(0.35 * len(models) + 2., 6.))
    _draw_tau_bar(
        ax, models, tau_map,
        rf'Phase-difference $\tau$ fit: CMIP6 models vs JRA55 -- {season}{title_suffix}',
        jra55_value=jra55_value
    )
    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


# ---------------------------------------------------------------------------
# Scatter of EFP vs tau, one panel per spatial scale.
# ---------------------------------------------------------------------------
def _draw_efp_vs_tau_row(axes_row, models, tau_map, efp_map, jra55_point=None, title_suffix=''):
    for ax, key in zip(axes_row, efp_variants):
        x = np.array([efp_map[key][model] for model in models])
        y = np.array([tau_map[model] for model in models])

        for model, xi, yi in zip(models, x, y):
            ax.scatter(xi, yi, color=model_color_map[model], s=45,
                       edgecolor='k', linewidth=0.3, zorder=3)

        if len(x) > 1:
            m_coef, c_coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m_coef * xs + c_coef, color='k', lw=1., zorder=2)
            r, p = stats.pearsonr(x, y)
            p_str = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
            ax.text(0.95, 0.95, f'r={r:.2f}, {p_str}', transform=ax.transAxes,
                    fontsize=9, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))

        if jra55_point is not None and key in jra55_point:
            jx, jy = jra55_point[key]
            ax.scatter(jx, jy, marker='x', color='k', s=70, linewidth=2., zorder=4)

        if len(x) > 0:
            ax.scatter(x.mean(), y.mean(), marker='*', color='k', s=180,
                       edgecolor='k', linewidth=0.5, zorder=5)

        ax.axvline(0., color='0.3', lw=1.4, linestyle='--', zorder=1)
        ax.set_xlim(-0.05, 0.5)
        ax.set_xlabel('EFP', fontsize=10)
        ax.set_ylabel(r'$\tau$ (days)', fontsize=10)
        ax.set_title(f'{variant_titles[key]}{title_suffix}', fontsize=11)
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=9)


def _legend_handles(models, include_jra55=True):
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='k',
                   markeredgewidth=0.3, markerfacecolor=model_color_map[model],
                   markersize=6, label=model)
        for model in models
    ]
    if include_jra55:
        handles.append(
            plt.Line2D([0], [0], marker='x', color='k', linestyle='none',
                       markeredgewidth=2., markersize=8, label='JRA55')
        )
    handles.append(
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k',
                   markeredgecolor='k', markeredgewidth=0.5, markersize=12,
                   label='Multi-model mean')
    )
    return handles


def _plot_efp_vs_tau(models, out_file, tau_map, efp_map, jra55_point=None, title_suffix=''):
    fig, axes = plt.subplots(1, 3, figsize=(15., 5.), sharey=True)
    _draw_efp_vs_tau_row(axes, models, tau_map, efp_map, jra55_point=jra55_point,
                          title_suffix=title_suffix)

    fig.legend(handles=_legend_handles(models, include_jra55=jra55_point is not None),
               loc='lower center', ncol=7, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0., 0.08, 1., 1.))

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


def _plot_efp_vs_tau_stacked(models, outlier_models, out_file, tau_map, efp_map, jra55_point=None):
    models_no_outlier = [model for model in models if model not in outlier_models]

    fig, axes = plt.subplots(2, 3, figsize=(15., 10.), sharey=True)
    _draw_efp_vs_tau_row(axes[0], models, tau_map, efp_map, jra55_point=jra55_point)
    _draw_efp_vs_tau_row(axes[1], models_no_outlier, tau_map, efp_map, jra55_point=jra55_point,
                          title_suffix=_excluded_suffix(outlier_models))

    fig.text(0.005, 0.75, 'All models', fontsize=11, fontweight='bold',
              rotation=90, va='center', ha='center')
    fig.text(0.005, 0.28, 'Outlier excluded', fontsize=11, fontweight='bold',
              rotation=90, va='center', ha='center')

    fig.legend(handles=_legend_handles(models, include_jra55=jra55_point is not None),
               loc='lower center', ncol=7, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0.02, 0.06, 1., 1.))

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


# ---------------------------------------------------------------------------
# Per-season output: tau bar chart + EFP-vs-tau scatter (main/outlier/stacked).
# ---------------------------------------------------------------------------
for season in SEASONS:
    season_tag = season.lower()
    season_dir = os.path.join(subplot_dir, season)
    os.makedirs(season_dir, exist_ok=True)
    season_outlier_dirs = {}
    for tag, _ in OUTLIER_SETS:
        d = os.path.join(season_dir, tag)
        os.makedirs(d, exist_ok=True)
        season_outlier_dirs[tag] = d

    tau_map = tau_maps[season]
    efp_map = efp_maps[season]
    jra55_tau = jra55_tau_maps[season]
    jra55_point = jra55_points[season]
    season_scatter_models = scatter_models[season]

    _plot_tau_bar(used_models, tau_map, jra55_tau, season,
                  os.path.join(season_dir, 'CMIP6_fig6_tau_bar.png'))
    _plot_efp_vs_tau(season_scatter_models, os.path.join(season_dir, 'CMIP6_fig6_efp_vs_tau.png'),
                      tau_map, efp_map, jra55_point=jra55_point)

    for tag, outlier_models in OUTLIER_SETS:
        _plot_tau_bar(
            [model for model in used_models if model not in outlier_models],
            tau_map, jra55_tau, season,
            os.path.join(season_outlier_dirs[tag], 'CMIP6_fig6_tau_bar.png'),
            title_suffix=_excluded_suffix(outlier_models)
        )
        _plot_efp_vs_tau(
            [model for model in season_scatter_models if model not in outlier_models],
            os.path.join(season_outlier_dirs[tag], 'CMIP6_fig6_efp_vs_tau.png'),
            tau_map, efp_map, jra55_point=jra55_point,
            title_suffix=_excluded_suffix(outlier_models)
        )
        _plot_efp_vs_tau_stacked(
            season_scatter_models, outlier_models,
            os.path.join(season_dir, f'CMIP6_fig6_efp_vs_tau_stacked{_stacked_suffix(tag)}.png'),
            tau_map, efp_map, jra55_point=jra55_point
        )

# ---------------------------------------------------------------------------
# JJA-vs-DJF comparison: grouped bar chart (tau, two bars per model) with
# JRA55 drawn as two colour-matched reference lines.
# ---------------------------------------------------------------------------
SEASON_COLORS = {
    'JJA': '#DD8452',
    'DJF': '#55A868',
}


def _plot_tau_bar_jja_djf_grouped(models, out_file, title_suffix=''):
    group_labels = models + ['Multi-model\nmean']
    n_groups = len(group_labels)
    x_pos = np.arange(n_groups)
    bar_width = 0.35
    offsets = [-bar_width / 2., bar_width / 2.]

    fig, ax = plt.subplots(figsize=(0.5 * n_groups + 2., 6.5))

    for season, offset in zip(SEASONS, offsets):
        values = [tau_maps[season][model] for model in models]
        mmm = float(np.mean(values))
        values = values + [mmm]
        ax.bar(x_pos + offset, values, width=bar_width,
               color=SEASON_COLORS[season], label=season)

    for season in SEASONS:
        ax.axhline(
            jra55_tau_maps[season], color=SEASON_COLORS[season], lw=2., linestyle='--',
            label=rf'JRA55 {season} ($\tau$={jra55_tau_maps[season]:4.2f}d)'
        )

    ax.axvline(len(models) - 0.5, color='0.6', lw=0.8, linestyle=':', zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, rotation=90, fontsize=8)
    ax.set_ylabel(r'$\tau$ (days)', fontsize=11)
    ax.grid(True, axis='y')
    ax.legend(fontsize=8, frameon=True, loc='upper right', ncol=2)
    ax.set_title(rf'Phase-difference $\tau$ fit: JJA vs DJF{title_suffix}', fontsize=13)

    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_tau_bar_jja_djf_grouped(
    used_models, os.path.join(subplot_dir, 'CMIP6_fig6_tau_bar_jja_djf.png')
)
for _tag, _outlier_models in OUTLIER_SETS:
    _plot_tau_bar_jja_djf_grouped(
        [model for model in used_models if model not in _outlier_models],
        os.path.join(outlier_dirs[_tag], 'CMIP6_fig6_tau_bar_jja_djf.png'),
        title_suffix=_excluded_suffix(_outlier_models)
    )

# ---------------------------------------------------------------------------
# JJA-vs-DJF scatter comparison: two rows (JJA, DJF), three columns
# (spatial scale), panels labelled (a)-(f).
# ---------------------------------------------------------------------------
all_seasons_scatter_models = [
    model for model in used_models
    if all(model in scatter_models[season] for season in SEASONS)
]
print(f'Scattering JJA-vs-DJF EFP vs tau for {len(all_seasons_scatter_models)} model(s): '
      f'{", ".join(all_seasons_scatter_models)}')


def _plot_efp_vs_tau_jja_djf_stacked(models, out_file, title_suffix=''):
    fig, axes = plt.subplots(2, 3, figsize=(15., 10.), sharey='row')
    for axes_row, season in zip(axes, SEASONS):
        _draw_efp_vs_tau_row(axes_row, models, tau_maps[season], efp_maps[season],
                              jra55_point=jra55_points[season], title_suffix=title_suffix)

    panel_labels = 'abcdef'
    for label, ax in zip(panel_labels, axes.flat):
        ax.set_title(rf'$\mathbf{{({label})}}$ {ax.get_title()}', fontsize=11)

    for season, axes_row in zip(SEASONS, axes):
        axes_row[0].text(-0.22, 1., season, transform=axes_row[0].transAxes,
                          fontsize=11, fontweight='bold', rotation=90, va='bottom', ha='center')

    fig.legend(handles=_legend_handles(models, include_jra55=True),
               loc='lower center', ncol=7, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0.02, 0.06, 1., 1.))

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_efp_vs_tau_jja_djf_stacked(
    all_seasons_scatter_models, os.path.join(subplot_dir, 'CMIP6_fig6_efp_vs_tau_jja_djf.png')
)
for _tag, _outlier_models in OUTLIER_SETS:
    _plot_efp_vs_tau_jja_djf_stacked(
        [model for model in all_seasons_scatter_models if model not in _outlier_models],
        os.path.join(outlier_dirs[_tag], 'CMIP6_fig6_efp_vs_tau_jja_djf.png'),
        title_suffix=_excluded_suffix(_outlier_models)
    )

# ---------------------------------------------------------------------------
# Pearson r/p summary: every season x EFP-variant x model-exclusion set.
# ---------------------------------------------------------------------------
model_set_defs = [([], 'none')] + [(models, ', '.join(models)) for _, models in OUTLIER_SETS]

corr_rows = []
for season in SEASONS:
    tau_map = tau_maps[season]
    efp_map = efp_maps[season]
    base_models = scatter_models[season]
    for excluded_models, excluded_label in model_set_defs:
        set_models = [m for m in base_models if m not in excluded_models]
        for variant_key in efp_variants:
            x = np.array([efp_map[variant_key][m] for m in set_models])
            y = np.array([tau_map[m] for m in set_models])
            r, p = stats.pearsonr(x, y)
            corr_rows.append((season, variant_key, excluded_label, len(set_models), r, p))

corr_csv_file = os.path.join(subplot_dir, 'CMIP6_fig6_efp_vs_tau_correlations.csv')
with open(corr_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['season', 'variant', 'excluded_models', 'n', 'r', 'p'])
    for season, variant_key, excluded_label, n, r, p in corr_rows:
        writer.writerow([season, variant_key, excluded_label, n, f'{r:.4f}', f'{p:.4g}'])
print(f'Saved {corr_csv_file}')
