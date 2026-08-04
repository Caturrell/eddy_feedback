"""
Companion to CMIP6_fig5_tau_bar.py: identical design (per-time-frame tau bar
charts, b-vs-tau scatter panels for three spatial/wavenumber scales, JRA55
reference, cumulative outlier-exclusion sets, all-time/JJA/DJF comparison
figures, Pearson r/p summary CSV) but uses the "native" wind variant
throughout instead of "va" (vertically-averaged 250/500/850hPa ucomp).

JRA55 native tau is available (jra55_tau_fit_3.csv has wind_variant=native
rows for all_time/JJA/DJF). JRA55 native b comes from
b-parameter/cmip6_b/efp_vs_b/jra55_b_native.csv (same row schema as the va
files -- model,variant,hemisphere,season,b -- but a single file covering
both all_time and the 12 rolling seasons, unlike the va split across
jra55_b_all_time.csv / jra55_b_annual_cycle.csv). If that file is ever
missing, JRA55 is omitted from every b-vs-tau scatter panel (crosses,
regression exclusion, and legend entry all skip it) but still appears as a
reference line on the tau bar charts, since native tau is available.

Models are sorted alphabetically (matching the model legend in
CMIP6_fig1_cospec_coher_pdiff.py and CMIP6_fig5_tau_bar.py) and use the same
turbo colormap colours.

Cumulative outlier-exclusion sets (see CMIP6_fig5_tau_bar.py for why):
  - remove_outlier: EC-Earth3-CC excluded (27 models)
  - remove_fgoals: EC-Earth3-CC AND FGOALS-f3-L excluded (26 models)

Output layout under plots/fig5native_tau_bar_b_vs_tau/:
  - all_time/ (and all_time/remove_outlier/, all_time/remove_fgoals/): the
    all-time bar chart and b-vs-tau scatter panels (main / stacked).
  - JJA/, DJF/, and NDJ/ (each with remove_outlier/ and remove_fgoals/
    subfolders): a combined bar chart stacking the all-time row above the
    seasonal row, plus b-vs-tau scatter panels (main / outlier-removed /
    stacked). NDJ is per-season only -- it is not folded into the
    all-time/JJA/DJF combined comparison figures below.
  - directly in fig5native_tau_bar_b_vs_tau/ (and its own remove_outlier/,
    remove_fgoals/): CMIP6_fig5native_tau_bar_all_seasons.png (single-panel
    grouped bar chart, three bars per model/mean, one colour per time frame,
    JRA55 drawn as three matching reference lines).
  - directly in plots/ (baseline only): CMIP6_fig5native_b_vs_tau_all_seasons.png
    (three scatter rows, one per time frame, each sharing its row's tau
    axis, panels labelled (a)-(i)); the remove_outlier/remove_fgoals
    versions instead save inside fig5native_tau_bar_b_vs_tau/.
  - CMIP6_fig5native_b_vs_tau_correlations.csv: Pearson r/p for every
    (time frame x b-variant x model-exclusion set) combination.

Tau source data: b-parameter/b_methodology/all_plots_true/250-500-850hPa_dm/
                 1979_2015/<model>/6hrPlevPt/power_spec.nc
JRA55 tau reference data: b-parameter/b_methodology/tau_values/data/jra55_tau_fit_3.csv
b source data: b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/
               <model>/6hrPlevPt/b_dataset.nc
JRA55 b reference data (native): b-parameter/cmip6_b/efp_vs_b/jra55_b_native.csv
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

b_base_dir = os.path.normpath(os.path.join(
    script_dir, '..', '..', 'cmip6_b', '250-500-850hPa_dm', '1979_2015'
))

hemisphere = 's'

b_variants = {
    'full': 'div1_QG',
    'k1-3': 'div1_QG_123',
    'gt3': 'div1_QG_gt3',
}

variant_titles = {
    'full': 'All Wavenumbers',
    'k1-3': 'Wavenumbers 1-3',
    'gt3': 'Wavenumbers >3',
}

# Cumulative: each set excludes everything the previous set(s) excluded, plus
# its own model(s) -- 'remove_outlier' drops EC-Earth3-CC (27 models),
# 'remove_fgoals' drops EC-Earth3-CC AND FGOALS-f3-L on top of that (26).
OUTLIER_SETS = [
    ('remove_outlier', ['EC-Earth3-CC']),
    ('remove_fgoals', ['EC-Earth3-CC', 'FGOALS-f3-L']),
]


def _stacked_suffix(tag):
    """Filename suffix for 'stacked' (all-models + outlier-excluded row/column in
    one file) comparisons: '' for the first/default outlier set, '_<name>' after."""
    return '' if tag == 'remove_outlier' else f'_{tag.split("_", 1)[1]}'


def _excluded_suffix(outlier_models):
    return f' ({", ".join(outlier_models)} excluded)'


model_names = sorted(
    d for d in os.listdir(cmip6_base_dir)
    if os.path.isdir(os.path.join(cmip6_base_dir, d))
)


def _load_tau_data(time_frame):
    ucomp_name = f'ucomp_PCs_{hemisphere}_{time_frame}'
    div1_name = f'div1_QG_PCs_from_ucomp_{hemisphere}_{time_frame}'
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


def _load_b_data(models, time_frame):
    data = {key: {} for key in b_variants}
    skipped = []
    for model in models:
        b_file = os.path.join(b_base_dir, model, '6hrPlevPt', 'b_dataset.nc')
        if not os.path.isfile(b_file):
            skipped.append(model)
            continue
        with xar.open_dataset(b_file) as b_ds:
            try:
                for key, var_to_analyse in b_variants.items():
                    b_var = f'ucomp_{var_to_analyse}_b_{hemisphere}_{time_frame}'
                    data[key][model] = float(b_ds[b_var].mean('lag', skipna=True))
            except KeyError:
                skipped.append(model)
                for key in b_variants:
                    data[key].pop(model, None)
    if skipped:
        print(f'Warning: skipped {len(skipped)} model(s) with no usable b_dataset.nc '
              f'for b ({time_frame}): {", ".join(skipped)}')
    return data


def _load_jra55_b(csv_path, season):
    values = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row['hemisphere'] != hemisphere or row['season'] != season:
                continue
            for key, var_to_analyse in b_variants.items():
                if row['variant'] == var_to_analyse:
                    values[key] = float(row['b'])
    return values


def _load_jra55_tau(csv_path, season, wind_variant='native'):
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if (row['hemisphere'] == hemisphere and row['season'] == season
                    and row['wind_variant'] == wind_variant and row['qg_mode'] == 'div1_QG'):
                return float(row['tau_fit_3'])
    raise ValueError(
        f'No JRA55 tau found for season={season}, hemisphere={hemisphere}, '
        f'wind_variant={wind_variant} in {csv_path}'
    )


# JRA55 native b: single file covering both all_time and the 12 rolling
# seasons. Detected at import time so every downstream plot can gracefully
# omit the JRA55 marker if it's ever missing, with no code changes needed.
JRA55_B_FILE = os.path.normpath(
    os.path.join(script_dir, '..', '..', 'cmip6_b', 'efp_vs_b', 'jra55_b_native.csv')
)
JRA55_NATIVE_B_AVAILABLE = os.path.isfile(JRA55_B_FILE)
if not JRA55_NATIVE_B_AVAILABLE:
    print(f'Warning: JRA55 native b not found at {JRA55_B_FILE}\n'
          'JRA55 will be omitted from all b-vs-tau scatter panels (tau bar-chart '
          'reference lines are unaffected, since native tau is available).')


# ---------------------------------------------------------------------------
# All-time data
# ---------------------------------------------------------------------------
tau_data = _load_tau_data('all_time')
used_models = sorted(tau_data)
print(f'Plotting {len(used_models)} model(s): {", ".join(used_models)}')

jra55_tau_file = os.path.normpath(
    os.path.join(script_dir, '..', 'tau_values', 'data', 'jra55_tau_fit_3.csv')
)
jra55_tau = _load_jra55_tau(jra55_tau_file, 'all_time')
print(f'Loaded JRA55 all-time tau from {jra55_tau_file}: {jra55_tau:.4f}d')

# Same colormap/order as CMIP6_fig1_cospec_coher_pdiff.py so colours match
# the legend there.
n_models = len(used_models)
model_colors = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, n_models))
model_color_map = dict(zip(used_models, model_colors))

tau_values = [tau_data[model] for model in used_models]
mmm_tau = float(np.mean(tau_values))

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
csv_file = os.path.join(data_dir, 'CMIP6_tau_fit_3_native.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'tau_fit_3_days'])
    for model in used_models:
        writer.writerow([model, f'{tau_data[model]:.4f}'])
    writer.writerow(['Multi-model mean', f'{mmm_tau:.4f}'])
    writer.writerow(['JRA55', f'{jra55_tau:.4f}'])
print(f'Saved {csv_file}')

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)
subplot_dir = os.path.join(plot_dir, 'fig5native_tau_bar_b_vs_tau')
os.makedirs(subplot_dir, exist_ok=True)
outlier_dirs = {}
for _tag, _ in OUTLIER_SETS:
    _d = os.path.join(subplot_dir, _tag)
    os.makedirs(_d, exist_ok=True)
    outlier_dirs[_tag] = _d

all_time_dir = os.path.join(subplot_dir, 'all_time')
os.makedirs(all_time_dir, exist_ok=True)
all_time_outlier_dirs = {}
for _tag, _ in OUTLIER_SETS:
    _d = os.path.join(all_time_dir, _tag)
    os.makedirs(_d, exist_ok=True)
    all_time_outlier_dirs[_tag] = _d

b_data = _load_b_data(used_models, 'all_time')

if JRA55_NATIVE_B_AVAILABLE:
    jra55_b = _load_jra55_b(JRA55_B_FILE, 'all_time')
    print(f'Loaded JRA55 all-time b from {JRA55_B_FILE}: {jra55_b}')
    jra55_point_all_time = {key: (jra55_b[key], jra55_tau) for key in jra55_b}
else:
    jra55_point_all_time = None

scatter_models = [
    model for model in used_models
    if all(model in b_data[key] for key in b_variants)
]
print(f'Scattering b vs tau for {len(scatter_models)} model(s): '
      f'{", ".join(scatter_models)}')

b_csv_file = os.path.join(data_dir, 'CMIP6_b_all_time_native.csv')
with open(b_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'b_full', 'b_k1-3', 'b_gt3'])
    for model in scatter_models:
        writer.writerow([
            model,
            f'{b_data["full"][model]:.4f}',
            f'{b_data["k1-3"][model]:.4f}',
            f'{b_data["gt3"][model]:.4f}',
        ])
print(f'Saved {b_csv_file}')


# ---------------------------------------------------------------------------
# Vertical bar chart, with a black multi-model-mean bar after the last model.
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
    ax.grid(True, axis='y')
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=True)
    ax.set_title(title, fontsize=13)


def _plot_tau_bar(models, out_file, title_suffix=''):
    fig, ax = plt.subplots(figsize=(0.35 * len(models) + 2., 6.))
    _draw_tau_bar(
        ax, models, tau_data,
        rf'Phase-difference $\tau$ fit (native): CMIP6 models vs JRA55{title_suffix}',
        jra55_value=jra55_tau
    )
    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


def _plot_tau_bar_seasons_stacked(models, season_tau_map, season_label, out_file,
                                   season_jra55_value=None, title_suffix=''):
    fig, axes = plt.subplots(2, 1, figsize=(0.35 * len(models) + 2., 11.))
    _draw_tau_bar(axes[0], models, tau_data, f'All-time{title_suffix}', jra55_value=jra55_tau)
    _draw_tau_bar(axes[1], models, season_tau_map, f'{season_label}{title_suffix}',
                  jra55_value=season_jra55_value)
    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_tau_bar(used_models, os.path.join(all_time_dir, 'CMIP6_fig5native_tau_bar.png'))
for _tag, _outlier_models in OUTLIER_SETS:
    _plot_tau_bar(
        [model for model in used_models if model not in _outlier_models],
        os.path.join(all_time_outlier_dirs[_tag], 'CMIP6_fig5native_tau_bar.png'),
        title_suffix=_excluded_suffix(_outlier_models)
    )

# ---------------------------------------------------------------------------
# Scatter of b vs tau, one panel per spatial scale.
# ---------------------------------------------------------------------------
def _draw_b_vs_tau_row(axes_row, models, tau_map, b_map, jra55_point=None, title_suffix='',
                        row_label=None, ylim=None):
    for ax, key in zip(axes_row, b_variants):
        x = np.array([b_map[key][model] for model in models])
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
        ax.set_xlim(-0.2, 0.2)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel('b', fontsize=10)
        ax.set_ylabel(r'$\tau$ (days)', fontsize=10)
        ax.set_title(f'{variant_titles[key]}{title_suffix}', fontsize=11)
        if row_label:
            ax.text(0.05, 0.95, row_label, transform=ax.transAxes, fontsize=10,
                    fontweight='bold', va='top', ha='left')
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


def _plot_b_vs_tau(models, out_file, tau_map, b_map, jra55_point=None, title_suffix=''):
    fig, axes = plt.subplots(1, 3, figsize=(15., 5.), sharey=True)
    _draw_b_vs_tau_row(axes, models, tau_map, b_map, jra55_point=jra55_point,
                        title_suffix=title_suffix)

    fig.legend(handles=_legend_handles(models, include_jra55=jra55_point is not None),
               loc='lower center', ncol=7, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0., 0.08, 1., 1.))

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


def _plot_b_vs_tau_stacked(models, outlier_models, out_file, tau_map, b_map, jra55_point=None):
    models_no_outlier = [model for model in models if model not in outlier_models]

    fig, axes = plt.subplots(2, 3, figsize=(15., 10.), sharey=True)
    _draw_b_vs_tau_row(axes[0], models, tau_map, b_map, jra55_point=jra55_point)
    _draw_b_vs_tau_row(axes[1], models_no_outlier, tau_map, b_map, jra55_point=jra55_point,
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


_plot_b_vs_tau(scatter_models, os.path.join(all_time_dir, 'CMIP6_fig5native_b_vs_tau.png'),
               tau_data, b_data, jra55_point=jra55_point_all_time)
for _tag, _outlier_models in OUTLIER_SETS:
    _plot_b_vs_tau(
        [model for model in scatter_models if model not in _outlier_models],
        os.path.join(all_time_outlier_dirs[_tag], 'CMIP6_fig5native_b_vs_tau.png'),
        tau_data, b_data, jra55_point=jra55_point_all_time,
        title_suffix=_excluded_suffix(_outlier_models)
    )
    _plot_b_vs_tau_stacked(
        scatter_models, _outlier_models,
        os.path.join(all_time_dir, f'CMIP6_fig5native_b_vs_tau_stacked{_stacked_suffix(_tag)}.png'),
        tau_data, b_data, jra55_point=jra55_point_all_time
    )

# ---------------------------------------------------------------------------
# Per-season (JJA, DJF) data + figures, all reusing the helpers above.
# ---------------------------------------------------------------------------
def _process_season(season):
    season_tau_data = _load_tau_data(season)
    season_b_data = _load_b_data(used_models, season)

    jra55_tau_season = _load_jra55_tau(jra55_tau_file, season)
    print(f'Loaded JRA55 {season} tau from {jra55_tau_file}: {jra55_tau_season:.4f}d')

    if JRA55_NATIVE_B_AVAILABLE:
        jra55_b_season = _load_jra55_b(JRA55_B_FILE, season)
        print(f'Loaded JRA55 {season} b from {JRA55_B_FILE}: {jra55_b_season}')
        jra55_point_season = {key: (jra55_b_season[key], jra55_tau_season) for key in jra55_b_season}
    else:
        jra55_point_season = None

    season_scatter_models = [
        model for model in used_models
        if model in season_tau_data and all(model in season_b_data[key] for key in b_variants)
    ]
    print(f'Scattering {season} b vs tau for {len(season_scatter_models)} model(s): '
          f'{", ".join(season_scatter_models)}')

    season_tag = season.lower()
    tau_csv_file = os.path.join(data_dir, f'CMIP6_tau_fit_3_{season_tag}_native.csv')
    with open(tau_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'tau_fit_3_days'])
        for model in used_models:
            writer.writerow([model, f'{season_tau_data[model]:.4f}'])
        writer.writerow(['Multi-model mean',
                          f'{float(np.mean([season_tau_data[m] for m in used_models])):.4f}'])
    print(f'Saved {tau_csv_file}')

    b_csv_file_season = os.path.join(data_dir, f'CMIP6_b_{season_tag}_native.csv')
    with open(b_csv_file_season, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'b_full', 'b_k1-3', 'b_gt3'])
        for model in season_scatter_models:
            writer.writerow([
                model,
                f'{season_b_data["full"][model]:.4f}',
                f'{season_b_data["k1-3"][model]:.4f}',
                f'{season_b_data["gt3"][model]:.4f}',
            ])
    print(f'Saved {b_csv_file_season}')

    season_dir = os.path.join(subplot_dir, season)
    os.makedirs(season_dir, exist_ok=True)
    season_outlier_dirs = {}
    for tag, _ in OUTLIER_SETS:
        d = os.path.join(season_dir, tag)
        os.makedirs(d, exist_ok=True)
        season_outlier_dirs[tag] = d

    _plot_tau_bar_seasons_stacked(
        used_models, season_tau_data, season,
        os.path.join(season_dir, f'CMIP6_fig5native_tau_bar_all_time_{season_tag}.png'),
        season_jra55_value=jra55_tau_season
    )
    _plot_b_vs_tau(season_scatter_models, os.path.join(season_dir, 'CMIP6_fig5native_b_vs_tau.png'),
                   season_tau_data, season_b_data, jra55_point=jra55_point_season)

    for tag, outlier_models in OUTLIER_SETS:
        _plot_tau_bar_seasons_stacked(
            [model for model in used_models if model not in outlier_models],
            season_tau_data, season,
            os.path.join(season_outlier_dirs[tag], f'CMIP6_fig5native_tau_bar_all_time_{season_tag}.png'),
            season_jra55_value=jra55_tau_season,
            title_suffix=_excluded_suffix(outlier_models)
        )
        _plot_b_vs_tau(
            [model for model in season_scatter_models if model not in outlier_models],
            os.path.join(season_outlier_dirs[tag], 'CMIP6_fig5native_b_vs_tau.png'),
            season_tau_data, season_b_data, jra55_point=jra55_point_season,
            title_suffix=_excluded_suffix(outlier_models)
        )
        _plot_b_vs_tau_stacked(
            season_scatter_models, outlier_models,
            os.path.join(season_dir, f'CMIP6_fig5native_b_vs_tau_stacked{_stacked_suffix(tag)}.png'),
            season_tau_data, season_b_data, jra55_point=jra55_point_season
        )

    return season_tau_data, season_b_data, jra55_tau_season, jra55_point_season


season_tau_maps = {}
season_b_maps = {}
season_jra55_tau = {}
season_jra55_points = {}
for _season in ['JJA', 'DJF', 'NDJ']:
    (season_tau_maps[_season], season_b_maps[_season],
     season_jra55_tau[_season], season_jra55_points[_season]) = _process_season(_season)

# ---------------------------------------------------------------------------
# Combined grouped bar chart: one panel, three bars per model (all-time,
# JJA, DJF) plus a multi-model-mean group, with JRA55 drawn as three
# reference lines (one per season, colour-matched to that season's bars).
# ---------------------------------------------------------------------------
SEASON_COLORS = {
    'All-time': '#4C72B0',
    'JJA': '#DD8452',
    'DJF': '#55A868',
}


def _plot_tau_bar_all_seasons_grouped(models, out_file, title_suffix=''):
    season_maps = {
        'All-time': tau_data,
        'JJA': season_tau_maps['JJA'],
        'DJF': season_tau_maps['DJF'],
    }
    season_jra55 = {
        'All-time': jra55_tau,
        'JJA': season_jra55_tau['JJA'],
        'DJF': season_jra55_tau['DJF'],
    }

    group_labels = models + ['Multi-model\nmean']
    n_groups = len(group_labels)
    x_pos = np.arange(n_groups)
    bar_width = 0.26
    offsets = [-bar_width, 0., bar_width]

    fig, ax = plt.subplots(figsize=(0.5 * n_groups + 2., 6.5))

    for (season_label, season_map), offset in zip(season_maps.items(), offsets):
        values = [season_map[model] for model in models]
        mmm = float(np.mean(values))
        values = values + [mmm]
        ax.bar(x_pos + offset, values, width=bar_width,
               color=SEASON_COLORS[season_label], label=season_label)

    for season_label in season_maps:
        ax.axhline(
            season_jra55[season_label], color=SEASON_COLORS[season_label],
            lw=2., linestyle='--',
            label=rf'JRA55 {season_label} ($\tau$={season_jra55[season_label]:4.2f}d)'
        )

    ax.axvline(len(models) - 0.5, color='0.6', lw=0.8, linestyle=':', zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, rotation=90, fontsize=8)
    ax.set_ylabel(r'$\tau$ (days)', fontsize=11)
    ax.set_ylim(4., 18.5)
    ax.grid(True, axis='y')
    ax.legend(fontsize=8, frameon=True, loc='upper right', ncol=2)
    ax.set_title(
        rf'Phase-difference $\tau$ fit (native): All-time vs JJA vs DJF{title_suffix}', fontsize=13
    )

    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_tau_bar_all_seasons_grouped(
    used_models, os.path.join(subplot_dir, 'CMIP6_fig5native_tau_bar_all_seasons.png')
)
for _tag, _outlier_models in OUTLIER_SETS:
    _plot_tau_bar_all_seasons_grouped(
        [model for model in used_models if model not in _outlier_models],
        os.path.join(outlier_dirs[_tag], 'CMIP6_fig5native_tau_bar_all_seasons.png'),
        title_suffix=_excluded_suffix(_outlier_models)
    )

# ---------------------------------------------------------------------------
# Combined scatter stacking all-time, JJA, and DJF b-vs-tau in one figure
# (three rows, each sharing its own tau axis across the three variant
# columns), each row with its own JRA55 cross (when available).
# ---------------------------------------------------------------------------
all_seasons_scatter_models = [
    model for model in scatter_models
    if all(model in season_tau_maps[s] for s in ('JJA', 'DJF'))
    and all(model in season_b_maps[s][key] for s in ('JJA', 'DJF') for key in b_variants)
]
print(f'Scattering all-seasons b vs tau for {len(all_seasons_scatter_models)} model(s): '
      f'{", ".join(all_seasons_scatter_models)}')


def _plot_b_vs_tau_all_seasons_stacked(models, out_file, title_suffix=''):
    fig, axes = plt.subplots(3, 3, figsize=(15., 15.), sharey='row')
    _draw_b_vs_tau_row(axes[0], models, tau_data, b_data,
                        jra55_point=jra55_point_all_time, title_suffix=title_suffix,
                        row_label='All-time', ylim=(4., 18.5))
    _draw_b_vs_tau_row(axes[1], models, season_tau_maps['JJA'], season_b_maps['JJA'],
                        jra55_point=season_jra55_points['JJA'], title_suffix=title_suffix,
                        row_label='JJA', ylim=(4., 18.5))
    _draw_b_vs_tau_row(axes[2], models, season_tau_maps['DJF'], season_b_maps['DJF'],
                        jra55_point=season_jra55_points['DJF'], title_suffix=title_suffix,
                        row_label='DJF', ylim=(4., 18.5))

    panel_labels = 'abcdefghi'
    for label, ax in zip(panel_labels, axes.flat):
        ax.set_title(rf'$\mathbf{{({label})}}$ {ax.get_title()}', fontsize=11)

    include_jra55 = jra55_point_all_time is not None
    fig.legend(handles=_legend_handles(models, include_jra55=include_jra55),
               loc='lower center', ncol=7, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0., 0.05, 1., 1.))

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_b_vs_tau_all_seasons_stacked(
    all_seasons_scatter_models, os.path.join(plot_dir, 'CMIP6_fig5native_b_vs_tau_all_seasons.png')
)
for _tag, _outlier_models in OUTLIER_SETS:
    _plot_b_vs_tau_all_seasons_stacked(
        [model for model in all_seasons_scatter_models if model not in _outlier_models],
        os.path.join(outlier_dirs[_tag], 'CMIP6_fig5native_b_vs_tau_all_seasons.png'),
        title_suffix=_excluded_suffix(_outlier_models)
    )

# ---------------------------------------------------------------------------
# Pearson r/p summary: every time frame x b-variant x model-exclusion set
# (all models, EC-Earth3-CC excluded, EC-Earth3-CC + FGOALS-f3-L excluded).
# ---------------------------------------------------------------------------
model_set_defs = [([], 'none')] + [(models, ', '.join(models)) for _, models in OUTLIER_SETS]

tau_maps_by_time = {'All-time': tau_data, 'JJA': season_tau_maps['JJA'], 'DJF': season_tau_maps['DJF']}
b_maps_by_time = {'All-time': b_data, 'JJA': season_b_maps['JJA'], 'DJF': season_b_maps['DJF']}
scatter_models_by_time = {
    'All-time': scatter_models,
    'JJA': [m for m in used_models
            if m in season_tau_maps['JJA'] and all(m in season_b_maps['JJA'][k] for k in b_variants)],
    'DJF': [m for m in used_models
            if m in season_tau_maps['DJF'] and all(m in season_b_maps['DJF'][k] for k in b_variants)],
}

corr_rows = []
for time_label, tau_map in tau_maps_by_time.items():
    b_map = b_maps_by_time[time_label]
    base_models = scatter_models_by_time[time_label]
    for excluded_models, excluded_label in model_set_defs:
        set_models = [m for m in base_models if m not in excluded_models]
        for variant_key in b_variants:
            x = np.array([b_map[variant_key][m] for m in set_models])
            y = np.array([tau_map[m] for m in set_models])
            r, p = stats.pearsonr(x, y)
            corr_rows.append((time_label, variant_key, excluded_label, len(set_models), r, p))

corr_csv_file = os.path.join(subplot_dir, 'CMIP6_fig5native_b_vs_tau_correlations.csv')
with open(corr_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time_frame', 'variant', 'excluded_models', 'n', 'r', 'p'])
    for time_label, variant_key, excluded_label, n, r, p in corr_rows:
        writer.writerow([time_label, variant_key, excluded_label, n, f'{r:.4f}', f'{p:.4g}'])
print(f'Saved {corr_csv_file}')
