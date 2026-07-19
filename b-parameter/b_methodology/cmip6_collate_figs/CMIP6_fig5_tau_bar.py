"""
Vertical bar chart of the phase-difference tau fit (tau_fit_3) for each
CMIP6 model, plus a black reference line for JRA55's tau value. A black bar
for the multi-model mean is added after the last (alphabetically) model.

Also scatters the all-time b-parameter (Simpson et al. 2013 lag-regression
method, southern hemisphere, vertically averaged over 250/500/850hPa)
against tau for each model, for three spatial (wavenumber) scales: full
spectrum, wavenumbers 1-3, and wavenumbers >3. No JRA55 reference is plotted
on the scatter panels since no all-time b has been computed for JRA55.

Models are sorted alphabetically (matching the model legend in
CMIP6_fig1_cospec_coher_pdiff.py) and use the same turbo colormap colours
throughout.

Per-model tau values, the multi-model mean, and JRA55's tau are saved to
data/CMIP6_tau_fit_3.csv. Per-model b values (all three variants) are saved
to data/CMIP6_b_all_time.csv.

Both figures are also regenerated with EC-Earth3-CC (a heavy outlier in tau)
excluded, saved to plots/fig5_tau_bar_b_vs_tau/remove_outlier/.

Tau source data: b-parameter/b_methodology/all_plots_true/250-500-850hPa_dm/
                 1979_2015/<model>/6hrPlevPt/power_spec.nc
JRA55 tau reference data: b-parameter/b_methodology/data/cospec_coher_pdiff_jra55.npz
b source data: b-parameter/cmip6_b/250-500-850hPa_dm/1979_2015/
               <model>/6hrPlevPt/b_dataset.nc
"""

import os
import csv
import numpy as np
import xarray as xar
import matplotlib.pyplot as plt
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))

cmip6_base_dir = os.path.join(
    script_dir, '..', 'all_plots_true', '250-500-850hPa_dm', '1979_2015'
)
cmip6_base_dir = os.path.normpath(cmip6_base_dir)

b_base_dir = os.path.join(
    script_dir, '..', '..', 'cmip6_b', '250-500-850hPa_dm', '1979_2015'
)
b_base_dir = os.path.normpath(b_base_dir)

hemisphere = 's'
time_frame = 'all_time'
time_name = 'time'

ucomp_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'
tau_var = f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'

model_names = sorted(
    d for d in os.listdir(cmip6_base_dir)
    if os.path.isdir(os.path.join(cmip6_base_dir, d))
)

tau_data = {}
skipped_models = []

for model in model_names:
    power_spec_file = os.path.join(
        cmip6_base_dir, model, '6hrPlevPt', 'power_spec.nc'
    )

    if not os.path.isfile(power_spec_file):
        skipped_models.append(model)
        continue

    with xar.open_dataset(power_spec_file, auto_complex=True) as power_spec_ds:
        try:
            tau_data[model] = float(power_spec_ds[tau_var])
        except KeyError:
            skipped_models.append(model)

if skipped_models:
    print(f'Warning: skipped {len(skipped_models)} model(s) with no usable '
          f'power_spec.nc: {", ".join(skipped_models)}')

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

OUTLIER_MODELS = ['EC-Earth3-CC']


# ---------------------------------------------------------------------------
# Vertical bar chart, with a black multi-model-mean bar after the last model.
# ---------------------------------------------------------------------------
def _plot_tau_bar(models, out_file, title_suffix=''):
    n = len(models)
    values = [tau_data[model] for model in models]
    mmm = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(0.35 * n + 2., 6.))

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

    h_jra55 = ax.axhline(
        jra55_tau, color='k', lw=2., linestyle='-',
        label=rf'JRA55 ($\tau$={jra55_tau:4.2f}d)'
    )

    ax.set_ylabel(r'$\tau$ (days)', fontsize=11)
    ax.set_ylim(0., max(bar_values) + 2.)
    ax.grid(True, axis='y')
    ax.legend(handles=[h_jra55], loc='upper right', fontsize=9, frameon=True)
    ax.set_title(
        rf'Phase-difference $\tau$ fit: CMIP6 models vs JRA55{title_suffix}',
        fontsize=13
    )

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
# b-parameter (all-time, southern hemisphere, vertically averaged) for three
# spatial scales, per model.
# ---------------------------------------------------------------------------
b_variants = {
    'full': 'div1_QG',
    'k1-3': 'div1_QG_123',
    'gt3': 'div1_QG_gt3',
}

b_data = {key: {} for key in b_variants}
b_skipped_models = []

for model in used_models:
    b_file = os.path.join(b_base_dir, model, '6hrPlevPt', 'b_dataset.nc')

    if not os.path.isfile(b_file):
        b_skipped_models.append(model)
        continue

    with xar.open_dataset(b_file) as b_ds:
        try:
            for key, var_to_analyse in b_variants.items():
                b_var = f'ucomp_va_{var_to_analyse}_va_b_{hemisphere}_{time_frame}'
                b_data[key][model] = float(b_ds[b_var].mean('lag', skipna=True))
        except KeyError:
            b_skipped_models.append(model)
            for key in b_variants:
                b_data[key].pop(model, None)

if b_skipped_models:
    print(f'Warning: skipped {len(b_skipped_models)} model(s) with no usable '
          f'b_dataset.nc: {", ".join(b_skipped_models)}')

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
# Scatter of b vs tau, one panel per spatial scale.
# ---------------------------------------------------------------------------
variant_titles = {
    'full': 'Full spectrum',
    'k1-3': 'Wavenumbers 1-3',
    'gt3': 'Wavenumbers >3',
}

def _draw_b_vs_tau_row(axes_row, models, title_suffix=''):
    for ax, key in zip(axes_row, b_variants):
        x = np.array([b_data[key][model] for model in models])
        y = np.array([tau_data[model] for model in models])

        for model, xi, yi in zip(models, x, y):
            ax.scatter(xi, yi, color=model_color_map[model], s=45,
                       edgecolor='k', linewidth=0.3, zorder=3)

        if len(x) > 1:
            m_coef, c_coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m_coef * xs + c_coef, color='k', lw=1., zorder=2)
            r, p = stats.pearsonr(x, y)
            p_str = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
            ax.text(0.05, 0.95, f'r={r:.2f}, {p_str}', transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))

        ax.axvline(0., color='0.6', lw=0.8, linestyle='--', zorder=1)
        ax.set_xlabel('b', fontsize=10)
        ax.set_ylabel(r'$\tau$ (days)', fontsize=10)
        ax.set_title(f'{variant_titles[key]}{title_suffix}', fontsize=11)
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=9)


def _legend_handles(models):
    return [
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='k',
                   markeredgewidth=0.3, markerfacecolor=model_color_map[model],
                   markersize=6, label=model)
        for model in models
    ]


def _plot_b_vs_tau(models, out_file, title_suffix=''):
    fig, axes = plt.subplots(1, 3, figsize=(15., 5.), sharey=True)
    _draw_b_vs_tau_row(axes, models, title_suffix=title_suffix)

    fig.legend(handles=_legend_handles(models), loc='lower center', ncol=7,
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0., 0.08, 1., 1.))

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


def _plot_b_vs_tau_stacked(models, outlier_models, out_file):
    models_no_outlier = [model for model in models if model not in outlier_models]

    fig, axes = plt.subplots(2, 3, figsize=(15., 10.), sharey=True)
    _draw_b_vs_tau_row(axes[0], models)
    _draw_b_vs_tau_row(axes[1], models_no_outlier,
                        title_suffix=f' ({outlier_models[0]} excluded)')

    fig.text(0.005, 0.75, 'All models', fontsize=11, fontweight='bold',
              rotation=90, va='center', ha='center')
    fig.text(0.005, 0.28, 'Outlier excluded', fontsize=11, fontweight='bold',
              rotation=90, va='center', ha='center')

    fig.legend(handles=_legend_handles(models), loc='lower center', ncol=7,
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0.02, 0.06, 1., 1.))

    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_plot_b_vs_tau(scatter_models, os.path.join(subplot_dir, 'CMIP6_fig5_b_vs_tau.png'))
_plot_b_vs_tau(
    [model for model in scatter_models if model not in OUTLIER_MODELS],
    os.path.join(outlier_dir, 'CMIP6_fig5_b_vs_tau.png'),
    title_suffix=f' ({OUTLIER_MODELS[0]} excluded)'
)
_plot_b_vs_tau_stacked(
    scatter_models, OUTLIER_MODELS,
    os.path.join(subplot_dir, 'CMIP6_fig5_b_vs_tau_stacked.png')
)
