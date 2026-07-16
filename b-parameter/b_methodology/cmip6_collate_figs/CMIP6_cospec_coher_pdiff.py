"""
Collates the cospectrum density, coherence-squared, and phase-difference
power-spectrum diagnostics (originally three separate PDFs produced by
functions.SIT_functions.SIT_eddy_feedback_functions.power_spectrum_analysis)
across all CMIP6 models into three grid figures - one panel per model.

Source data: b-parameter/b_methodology/all_plots_true/250-500-850hPa_dm/
             1979_2015/<model>/6hrPlevPt/power_spec.nc

Uses div1_QG (southern hemisphere, all_time, "va" variant), matching the
per-model plots in
all_plots_true/250-500-850hPa_dm/1979_2015/<model>/6hrPlevPt/
power_spec_plots/s/all_time/_va/

Note: a JRA55 reanalysis reference curve was previously overlaid on these
plots, but no JRA55 power_spec.nc currently exists anywhere in the repo, so
that reference has been dropped for now.
"""

import os
import numpy as np
import xarray as xar
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

cmip6_base_dir = os.path.join(
    script_dir, '..', 'all_plots_true', '250-500-850hPa_dm', '1979_2015'
)
cmip6_base_dir = os.path.normpath(cmip6_base_dir)

hemisphere = 's'
time_frame = 'all_time'
time_name = 'time'
frequency_name = f'frequency_{time_name}'

ucomp_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'

model_names = sorted(
    d for d in os.listdir(cmip6_base_dir)
    if os.path.isdir(os.path.join(cmip6_base_dir, d))
)

models_data = {}
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
            models_data[model] = {
                'freq': power_spec_ds[frequency_name].values,
                'cospec': power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft'].values,
                'coher': power_spec_ds[f'{ucomp_name}_{div1_name}_coher_stft'].values,
                'phase_diff': power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff'].values,
                'tau_fit_3': float(power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3']),
            }
        except KeyError:
            skipped_models.append(model)

if skipped_models:
    print(f'Warning: skipped {len(skipped_models)} model(s) with no usable '
          f'power_spec.nc: {", ".join(skipped_models)}')

used_models = sorted(models_data)
print(f'Plotting {len(used_models)} model(s): {", ".join(used_models)}')

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

n_models = len(used_models)
n_cols = 6
n_rows = int(np.ceil(n_models / n_cols))


def _make_grid_figure(suptitle):
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3. * n_cols, 2.5 * n_rows), sharex=True
    )
    axes = np.atleast_2d(axes)
    fig.suptitle(suptitle, fontsize=14)
    for extra_ax in axes.flat[n_models:]:
        extra_ax.axis('off')
    return fig, axes


def _finish_and_save(fig, axes, out_name, ylabel, legend_handles):
    for ax in axes.flat[:n_models]:
        ax.set_xlim(0., 0.25)
        ax.grid(True)
        ax.tick_params(labelsize=7)
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=8)
    # bottom-most active axis in each column gets the x-label
    active_ids = {id(ax) for ax in axes.flat[:n_models]}
    for col in range(n_cols):
        col_active = [ax for ax in axes[:, col] if id(ax) in active_ids]
        if col_active:
            col_active[-1].set_xlabel('frequency (1/days)', fontsize=8)

    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.tight_layout(rect=(0., 0.02, 1., 0.97))

    out_file = os.path.join(plot_dir, out_name)
    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


# ---------------------------------------------------------------------------
# Cospectrum grid
# ---------------------------------------------------------------------------
fig, axes = _make_grid_figure('Cospectrum of ucomp and div1 (all CMIP6 models)')
legend_handles = None
for ax, model in zip(axes.flat, used_models):
    data = models_data[model]
    freq = data['freq']
    (h_real,) = ax.plot(freq, np.real(data['cospec']), linestyle='--', label='real STFT')
    (h_imag,) = ax.plot(freq, np.imag(data['cospec']), linestyle='--', label='Imag STFT')
    (h_ref,) = ax.plot(freq, 2. * np.pi * freq, label='2piomega')
    ax.set_ylim(0., 1.75)
    ax.set_title(model, fontsize=8)
    if legend_handles is None:
        legend_handles = [h_real, h_imag, h_ref]
_finish_and_save(fig, axes, 'fig_cospec_grid.png', 'cospectrum', legend_handles)

# ---------------------------------------------------------------------------
# Coherence-squared grid
# ---------------------------------------------------------------------------
fig, axes = _make_grid_figure('Coherence squared of ucomp and div1 using stft method (all CMIP6 models)')
legend_handles = None
for ax, model in zip(axes.flat, used_models):
    data = models_data[model]
    freq = data['freq']
    (h_coher,) = ax.plot(freq, data['coher'] ** 2., linestyle='--', label='stft')
    ax.set_ylim(0., 1.)
    ax.set_title(model, fontsize=8)
    if legend_handles is None:
        legend_handles = [h_coher]
_finish_and_save(fig, axes, 'fig_coher_grid.png', 'coherence$^2$', legend_handles)

# ---------------------------------------------------------------------------
# Phase-difference grid
# ---------------------------------------------------------------------------
fig, axes = _make_grid_figure('Phase of ucomp and div1 using stft method (all CMIP6 models)')
legend_handles = None
for ax, model in zip(axes.flat, used_models):
    data = models_data[model]
    freq = data['freq']
    tau_fit_3 = data['tau_fit_3']
    (h_data,) = ax.plot(freq, data['phase_diff'], label='data')
    (h_fit,) = ax.plot(freq, np.rad2deg(np.arctan(2. * np.pi * freq * tau_fit_3)),
                        linestyle='--', color='#d62728', label=r'$\tau$ fit')
    ax.set_ylim(0., 90.)
    ax.set_title(model, fontsize=8)
    ax.text(0.95, 0.05, rf'$\tau$={tau_fit_3:4.2f}d', transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='right', va='bottom',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor='black', alpha=1.0))
    if legend_handles is None:
        legend_handles = [h_data, h_fit]
_finish_and_save(fig, axes, 'fig_phasediff_grid.png', 'phase diff (deg)', legend_handles)
