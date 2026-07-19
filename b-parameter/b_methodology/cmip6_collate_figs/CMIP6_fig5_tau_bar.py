"""
Vertical bar chart of the phase-difference tau fit (tau_fit_3) for each
CMIP6 model, plus a black reference line for JRA55's tau value. A black bar
for the multi-model mean is added after the last (alphabetically) model.

Also scatters the all-time b-parameter (Simpson et al. 2013 lag-regression
method, southern hemisphere, vertically averaged over 250/500/850hPa)
against tau for each model, for three spatial (wavenumber) scales: full
spectrum, wavenumbers 1-3, and wavenumbers >3. JRA55 is marked with a black
cross on each scatter panel, but is excluded from the regression line and
Pearson r/p annotation (those are CMIP6-models-only).

Models are sorted alphabetically (matching the model legend in
CMIP6_fig1_cospec_coher_pdiff.py) and use the same turbo colormap colours
throughout.

Per-model tau values, the multi-model mean, and JRA55's tau are saved to
data/CMIP6_tau_fit_3.csv. Per-model b values (all three variants) are saved
to data/CMIP6_b_all_time.csv.

All-time figures are also regenerated with EC-Earth3-CC (a heavy outlier in
tau) excluded, saved to plots/fig5_tau_bar_b_vs_tau/remove_outlier/.

A combined bar chart stacking all-time, JJA, and DJF tau (three rows, JRA55
omitted) is also saved directly to plots/fig5_tau_bar_b_vs_tau/ as
CMIP6_fig5_tau_bar_all_seasons.png, for both "with EC-Earth3-CC" and
"without" (remove_outlier/).

JJA and DJF versions are saved to plots/fig5_tau_bar_b_vs_tau/JJA/ and
.../DJF/ (each with its own remove_outlier/ subfolder):
  - a combined bar chart stacking the all-time row (as above) above the
    seasonal row, for both "with EC-Earth3-CC" and "without" -- JRA55 is
    omitted from this combined chart since no JRA55 seasonal tau exists
    anywhere in the repo.
  - b-vs-tau scatter panels (main / outlier-removed / stacked) using seasonal
    tau and seasonal b. JRA55 has a cached seasonal b
    (jra55_b_annual_cycle.csv) but, again, no seasonal tau, so it cannot be
    marked on these scatter panels.

Tau source data: b-parameter/b_methodology/all_plots_true/250-500-850hPa_dm/
                 1979_2015/<model>/6hrPlevPt/power_spec.nc
JRA55 tau reference data: b-parameter/b_methodology/data/cospec_coher_pdiff_jra55.npz
b source data: b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/
               <model>/6hrPlevPt/b_dataset.nc
JRA55 b reference data: b-parameter/cmip6_b/efp_vs_b/jra55_b_all_time.csv
                        b-parameter/cmip6_b/efp_vs_b/jra55_b_annual_cycle.csv (JJA/DJF)
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
    'full': 'Full spectrum',
    'k1-3': 'Wavenumbers 1-3',
    'gt3': 'Wavenumbers >3',
}

OUTLIER_MODELS = ['EC-Earth3-CC']

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
                    b_var = f'ucomp_va_{var_to_analyse}_va_b_{hemisphere}_{time_frame}'
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


# ---------------------------------------------------------------------------
# All-time data
# ---------------------------------------------------------------------------
tau_data = _load_tau_data('all_time')
used_models = sorted(tau_data)
print(f'Plotting {len(used_models)} model(s): {", ".join(used_models)}')

jra55_file = os.path.normpath(
    os.path.join(script_dir, '..', 'data', 'cospec_coher_pdiff_jra55.npz')
)
jra55_data = np.load(jra55_file)
jra55_tau = float(jra55_data['tau_fit_3'])

# Same colormap/order as CMIP6_fig1_cospec_coher_pdiff.py so colours match
# the legend there.
n_models = len(used_models)
model_colors = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, n_models))
model_color_map = dict(zip(used_models, model_colors))

tau_values = [tau_data[model] for model in used_models]
mmm_tau = float(np.mean(tau_values))

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
csv_file = os.path.join(data_dir, 'CMIP6_tau_fit_3.csv')
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
subplot_dir = os.path.join(plot_dir, 'fig5_tau_bar_b_vs_tau')
os.makedirs(subplot_dir, exist_ok=True)
outlier_dir = os.path.join(subplot_dir, 'remove_outlier')
os.makedirs(outlier_dir, exist_ok=True)

b_data = _load_b_data(used_models, 'all_time')

jra55_b_file = os.path.normpath(
    os.path.join(script_dir, '..', '..', 'cmip6_b', 'efp_vs_b', 'jra55_b_all_time.csv')
)
jra55_b = _load_jra55_b(jra55_b_file, 'all_time')
print(f'Loaded JRA55 all-time b from {jra55_b_file}: {jra55_b}')
jra55_point_all_time = {key: (jra55_b[key], jra55_tau) for key in jra55_b}

scatter_models = [
    model for model in used_models
    if all(model in b_data[key] for key in b_variants)
]
print(f'Scattering b vs tau for {len(scatter_models)} model(s): '
      f'{", ".join(scatter_models)}')

b_csv_file = os.path.join(data_dir, 'CMIP6_b_all_time.csv')
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
def _draw_tau_bar(ax, models, values_map, title, show_jra55=True):
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
    if show_jra55:
        h_jra55 = ax.axhline(
            jra55_tau, color='k', lw=2., linestyle='-',
            label=rf'JRA55 ($\tau$={jra55_tau:4.2f}d)'
        )
        handles.append(h_jra55)

    ax.set_ylabel(r'$\tau$ (days)', fontsize=11)
    ax.set_ylim(0., max(bar_values) + 2.)
    ax.grid(True, axis='y')
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=True)
    ax.set_title(title, fontsize=13)


def _plot_tau_bar(models, out_file, title_suffix=''):
    fig, ax = plt.subplots(figsize=(0.35 * len(models) + 2., 6.))
    _draw_tau_bar(
        ax, models, tau_data,
        rf'Phase-difference $\tau$ fit: CMIP6 models vs JRA55{title_suffix}',
        show_jra55=True
    )
    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


def _plot_tau_bar_seasons_stacked(models, season_tau_map, season_label, out_file, title_suffix=''):
    fig, axes = plt.subplots(2, 1, figsize=(0.35 * len(models) + 2., 11.))
    _draw_tau_bar(axes[0], models, tau_data, f'All-time{title_suffix}', show_jra55=False)
    _draw_tau_bar(axes[1], models, season_tau_map, f'{season_label}{title_suffix}', show_jra55=False)
    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_tau_bar(used_models, os.path.join(subplot_dir, 'CMIP6_fig5_tau_bar.png'))
_plot_tau_bar(
    [model for model in used_models if model not in OUTLIER_MODELS],
    os.path.join(outlier_dir, 'CMIP6_fig5_tau_bar.png'),
    title_suffix=f' ({OUTLIER_MODELS[0]} excluded)'
)

# ---------------------------------------------------------------------------
# Scatter of b vs tau, one panel per spatial scale.
# ---------------------------------------------------------------------------
def _draw_b_vs_tau_row(axes_row, models, tau_map, b_map, jra55_point=None, title_suffix=''):
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

        ax.axvline(0., color='0.6', lw=0.8, linestyle='--', zorder=1)
        ax.set_xlabel('b', fontsize=10)
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
                        title_suffix=f' ({outlier_models[0]} excluded)')

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


_plot_b_vs_tau(scatter_models, os.path.join(subplot_dir, 'CMIP6_fig5_b_vs_tau.png'),
               tau_data, b_data, jra55_point=jra55_point_all_time)
_plot_b_vs_tau(
    [model for model in scatter_models if model not in OUTLIER_MODELS],
    os.path.join(outlier_dir, 'CMIP6_fig5_b_vs_tau.png'),
    tau_data, b_data, jra55_point=jra55_point_all_time,
    title_suffix=f' ({OUTLIER_MODELS[0]} excluded)'
)
_plot_b_vs_tau_stacked(
    scatter_models, OUTLIER_MODELS,
    os.path.join(subplot_dir, 'CMIP6_fig5_b_vs_tau_stacked.png'),
    tau_data, b_data, jra55_point=jra55_point_all_time
)

# ---------------------------------------------------------------------------
# Per-season (JJA, DJF) data + figures, all reusing the helpers above.
# ---------------------------------------------------------------------------
jra55_b_annual_file = os.path.normpath(
    os.path.join(script_dir, '..', '..', 'cmip6_b', 'efp_vs_b', 'jra55_b_annual_cycle.csv')
)


def _process_season(season):
    season_tau_data = _load_tau_data(season)
    season_b_data = _load_b_data(used_models, season)

    jra55_b_season = _load_jra55_b(jra55_b_annual_file, season)
    print(f'Loaded JRA55 {season} b from {jra55_b_annual_file}: {jra55_b_season} '
          f'(no {season} tau available for JRA55, so not marked on {season} scatter panels)')

    season_scatter_models = [
        model for model in used_models
        if model in season_tau_data and all(model in season_b_data[key] for key in b_variants)
    ]
    print(f'Scattering {season} b vs tau for {len(season_scatter_models)} model(s): '
          f'{", ".join(season_scatter_models)}')

    season_tag = season.lower()
    tau_csv_file = os.path.join(data_dir, f'CMIP6_tau_fit_3_{season_tag}.csv')
    with open(tau_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'tau_fit_3_days'])
        for model in used_models:
            writer.writerow([model, f'{season_tau_data[model]:.4f}'])
        writer.writerow(['Multi-model mean',
                          f'{float(np.mean([season_tau_data[m] for m in used_models])):.4f}'])
    print(f'Saved {tau_csv_file}')

    b_csv_file_season = os.path.join(data_dir, f'CMIP6_b_{season_tag}.csv')
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
    season_outlier_dir = os.path.join(season_dir, 'remove_outlier')
    os.makedirs(season_outlier_dir, exist_ok=True)

    _plot_tau_bar_seasons_stacked(
        used_models, season_tau_data, season,
        os.path.join(season_dir, f'CMIP6_fig5_tau_bar_all_time_{season_tag}.png')
    )
    _plot_tau_bar_seasons_stacked(
        [model for model in used_models if model not in OUTLIER_MODELS],
        season_tau_data, season,
        os.path.join(season_outlier_dir, f'CMIP6_fig5_tau_bar_all_time_{season_tag}.png'),
        title_suffix=f' ({OUTLIER_MODELS[0]} excluded)'
    )

    _plot_b_vs_tau(season_scatter_models, os.path.join(season_dir, 'CMIP6_fig5_b_vs_tau.png'),
                   season_tau_data, season_b_data, jra55_point=None)
    _plot_b_vs_tau(
        [model for model in season_scatter_models if model not in OUTLIER_MODELS],
        os.path.join(season_outlier_dir, 'CMIP6_fig5_b_vs_tau.png'),
        season_tau_data, season_b_data, jra55_point=None,
        title_suffix=f' ({OUTLIER_MODELS[0]} excluded)'
    )
    _plot_b_vs_tau_stacked(
        season_scatter_models, OUTLIER_MODELS,
        os.path.join(season_dir, 'CMIP6_fig5_b_vs_tau_stacked.png'),
        season_tau_data, season_b_data, jra55_point=None
    )

    return season_tau_data


season_tau_maps = {}
for _season in ['JJA', 'DJF']:
    season_tau_maps[_season] = _process_season(_season)

# ---------------------------------------------------------------------------
# Combined bar chart stacking all-time, JJA, and DJF tau in one figure.
# JRA55 is omitted, as with the two-row all-time/season charts above.
# ---------------------------------------------------------------------------
def _plot_tau_bar_all_seasons_stacked(models, out_file, title_suffix=''):
    fig, axes = plt.subplots(3, 1, figsize=(0.35 * len(models) + 2., 16.))
    _draw_tau_bar(axes[0], models, tau_data, f'All-time{title_suffix}', show_jra55=False)
    _draw_tau_bar(axes[1], models, season_tau_maps['JJA'], f'JJA{title_suffix}', show_jra55=False)
    _draw_tau_bar(axes[2], models, season_tau_maps['DJF'], f'DJF{title_suffix}', show_jra55=False)
    fig.tight_layout()

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_tau_bar_all_seasons_stacked(
    used_models, os.path.join(subplot_dir, 'CMIP6_fig5_tau_bar_all_seasons.png')
)
_plot_tau_bar_all_seasons_stacked(
    [model for model in used_models if model not in OUTLIER_MODELS],
    os.path.join(outlier_dir, 'CMIP6_fig5_tau_bar_all_seasons.png'),
    title_suffix=f' ({OUTLIER_MODELS[0]} excluded)'
)
