"""
Collates the cospectrum density, coherence-squared, and phase-difference
power-spectrum diagnostics (originally three separate PDFs produced by
functions.SIT_functions.SIT_eddy_feedback_functions.power_spectrum_analysis)
across all CMIP6 models into grid figures (one panel per model) and spaghetti
figures (all models overlaid on one axes, plus the JRA55 reanalysis as a
thick black reference line).

The combined spaghetti figure also includes a SAM EOF1 panel, reusing the
per-model cache built by CMIP6_fig0_u_SAM_cmip6_hist.py.

Source data: b-parameter/b_methodology/all_plots_true/250-500-850hPa_dm/
             1979_2015/<model>/6hrPlevPt/power_spec.nc
JRA55 reference data: b-parameter/b_methodology/data/cospec_coher_pdiff_jra55.npz

Uses div1_QG (southern hemisphere, all_time, "va" variant), matching the
per-model plots in
all_plots_true/250-500-850hPa_dm/1979_2015/<model>/6hrPlevPt/
power_spec_plots/s/all_time/_va/
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

jra55_file = os.path.normpath(
    os.path.join(script_dir, '..', 'data', 'cospec_coher_pdiff_jra55.npz')
)
jra55_data = np.load(jra55_file)

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Individual grid/spaghetti figures go in a subdir; only the combined
# spaghetti figure is saved directly under plot_dir.
subplot_dir = os.path.join(plot_dir, 'fig1_cospec_coher_pdiff')
os.makedirs(subplot_dir, exist_ok=True)

n_models = len(used_models)
n_cols = 6
n_rows = int(np.ceil(n_models / n_cols))

model_colors = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, n_models))
model_color_map = dict(zip(used_models, model_colors))

# ---------------------------------------------------------------------------
# SAM EOF1 data (for the combined-figure SAM panel), reusing the per-model
# cache built by CMIP6_fig0_u_SAM_cmip6_hist.py.
# ---------------------------------------------------------------------------
sam_data_dir = os.path.normpath(
    os.path.join(script_dir, '..', '..', 'simpson_2013', 'data', 'sam_eofs')
)
sam_jra55_npz = np.load(os.path.join(sam_data_dir, 'sam_eof_250_500_850_jra55.npz'))
sam_jra55_lats = np.abs(sam_jra55_npz['lats'])
sam_jra55_eof1 = sam_jra55_npz['eof1']

sam_data = {}
sam_missing = []
for model in used_models:
    sam_cache_file = os.path.join(sam_data_dir, f'{model}_sam_eof_250_500_850.npz')
    if os.path.isfile(sam_cache_file):
        d = np.load(sam_cache_file)
        sam_data[model] = {'abs_lats': d['abs_lats'], 'eof1': d['eof1']}
    else:
        sam_missing.append(model)

if sam_missing:
    print(f'Warning: no cached SAM EOF1 for {len(sam_missing)} model(s) '
          f'(run CMIP6_fig0_u_SAM_cmip6_hist.py to populate the cache): '
          f'{", ".join(sam_missing)}')


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

    out_file = os.path.join(subplot_dir, out_name)
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
_finish_and_save(fig, axes, 'CMIP6_fig1_cospec_grid.png', 'cospectrum', legend_handles)

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
_finish_and_save(fig, axes, 'CMIP6_fig1_coher_grid.png', 'coherence$^2$', legend_handles)

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
_finish_and_save(fig, axes, 'CMIP6_fig1_phasediff_grid.png', 'phase diff (deg)', legend_handles)


# ---------------------------------------------------------------------------
# Spaghetti plots (all models overlaid, JRA55 as thick black reference)
# ---------------------------------------------------------------------------
jra55_legend_handle = plt.Line2D(
    [0], [0], color='k', lw=2.5,
    label=rf'JRA55 ($\tau$={float(jra55_data["tau_fit_3"]):4.2f}d)'
)
jra55_legend_handle_no_tau = plt.Line2D([0], [0], color='k', lw=2.5, label='JRA55')

model_legend_handles = [
    plt.Line2D([0], [0], color=color, lw=1.2,
               label=rf'{model} ($\tau$={models_data[model]["tau_fit_3"]:4.2f}d)')
    for model, color in zip(used_models, model_colors)
] + [jra55_legend_handle]
model_legend_handles_no_tau = [
    plt.Line2D([0], [0], color=color, lw=1.2, label=model)
    for model, color in zip(used_models, model_colors)
] + [jra55_legend_handle_no_tau]


def _plot_cospec_spaghetti(ax):
    for model, color in zip(used_models, model_colors):
        data = models_data[model]
        freq = data['freq']
        ax.plot(freq, np.real(data['cospec']), color=color, lw=0.8, alpha=0.7, linestyle='-')
        ax.plot(freq, np.imag(data['cospec']), color=color, lw=0.8, alpha=0.7, linestyle='--')
    ax.plot(jra55_data['freq'], np.real(jra55_data['cospec']),
            color='k', lw=2.5, linestyle='-')
    ax.plot(jra55_data['freq'], np.imag(jra55_data['cospec']),
            color='k', lw=2.5, linestyle='--')
    (h_ref,) = ax.plot(jra55_data['freq'], 2. * np.pi * jra55_data['freq'],
                        color='0.5', lw=2., label=r'$\omega = 2\pi f$')
    h_real_style = plt.Line2D([0], [0], color='k', lw=1.5, linestyle='-', label='real')
    h_imag_style = plt.Line2D([0], [0], color='k', lw=1.5, linestyle='--', label='imag')
    return ([h_real_style, h_imag_style, h_ref], 'cospectrum', (0., 1.75),
            (0., 0.25), 'frequency (1/days)')


def _plot_coher_spaghetti(ax):
    for model, color in zip(used_models, model_colors):
        data = models_data[model]
        ax.plot(data['freq'], data['coher'] ** 2., color=color, lw=0.8, alpha=0.7)
    ax.plot(jra55_data['freq'], jra55_data['coher'] ** 2., color='k', lw=2.5)
    return [], 'coherence$^2$', (0., 1.), (0., 0.25), 'frequency (1/days)'


def _plot_phasediff_spaghetti(ax):
    for model, color in zip(used_models, model_colors):
        data = models_data[model]
        ax.plot(data['freq'], data['phase_diff'], color=color, lw=0.8, alpha=0.7)
    ax.plot(jra55_data['freq'], jra55_data['phase_diff'], color='k', lw=2.5)
    return ([], 'phase diff (deg)', (0., 90.),
            (0., 0.25), 'frequency (1/days)')


def _plot_sam_spaghetti(ax):
    for model, data in sam_data.items():
        lats = -data['abs_lats']
        sort_idx = np.argsort(lats)
        ax.plot(lats[sort_idx], data['eof1'][sort_idx],
                color=model_color_map[model], lw=0.8, alpha=0.7)

    jra_lats = -sam_jra55_lats
    sort_jra = np.argsort(jra_lats)
    jra_lats_sorted = jra_lats[sort_jra]
    mmm_eof = np.nanmean([
        np.interp(jra_lats_sorted,
                  (-data['abs_lats'])[np.argsort(-data['abs_lats'])],
                  data['eof1'][np.argsort(-data['abs_lats'])])
        for data in sam_data.values()
    ], axis=0)

    (h_mmm,) = ax.plot(jra_lats_sorted, mmm_eof, color='k', lw=2.5, linestyle=':',
                        label='Multi-model mean')
    ax.plot(jra_lats_sorted, sam_jra55_eof1[sort_jra], color='k', lw=2.5)
    return ([h_mmm], 'EOF1 (m s$^{-1}$ PC$^{-1}$)', (-2.5, 4.5),
            (-80., -20.), 'latitude (°)')


def _decorate_spaghetti_axis(ax, ylabel, ylim, style_handles, xlim, xlabel):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(True)
    if style_handles:
        ax.legend(handles=style_handles, loc='upper left', fontsize=8, frameon=True)


def _make_single_spaghetti_figure(suptitle, plot_func, out_name):
    fig, ax = plt.subplots(figsize=(15., 6.5))
    fig_title = fig.suptitle(suptitle, fontsize=14)

    style_handles, ylabel, ylim, xlim, xlabel = plot_func(ax)
    _decorate_spaghetti_axis(ax, ylabel, ylim, style_handles, xlim, xlabel)

    fig.subplots_adjust(right=0.72, top=0.9)

    model_legend = ax.legend(
        handles=model_legend_handles, loc='upper left', bbox_to_anchor=(1.005, 1.02),
        fontsize=6, ncol=2, title='CMIP6 models', title_fontsize=7,
        frameon=False
    )
    ax.add_artist(model_legend)
    if style_handles:
        ax.legend(handles=style_handles, loc='upper left', fontsize=8, frameon=True)

    out_file = os.path.join(subplot_dir, out_name)
    fig.savefig(out_file, bbox_extra_artists=(model_legend, fig_title),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_make_single_spaghetti_figure(
    'Cospectrum of ucomp and div1 - CMIP6 models vs JRA55',
    _plot_cospec_spaghetti, 'CMIP6_fig1_cospec_spaghetti.png'
)
_make_single_spaghetti_figure(
    'Coherence squared of ucomp and div1 using stft method - CMIP6 models vs JRA55',
    _plot_coher_spaghetti, 'CMIP6_fig1_coher_spaghetti.png'
)
_make_single_spaghetti_figure(
    'Phase of ucomp and div1 using stft method - CMIP6 models vs JRA55',
    _plot_phasediff_spaghetti, 'CMIP6_fig1_phasediff_spaghetti.png'
)

# ---------------------------------------------------------------------------
# Combined spaghetti plot (SAM EOF1 + all 3 quantities as panels in one figure)
# ---------------------------------------------------------------------------
def _make_combined_figure(out_name, legend_handles, phasediff_plot_func):
    fig, axes = plt.subplots(2, 2, figsize=(16., 12.))

    panels = [
        (axes[0, 0], _plot_sam_spaghetti,
         r'$\mathbf{(a)}$ SAM EOF1 ($[\overline{u}]_s$)'),
        (axes[0, 1], _plot_cospec_spaghetti,
         r'$\mathbf{(b)}$ Cospectrum of $[\overline{u}]_s$ and $[\overline{m}]_s$'),
        (axes[1, 0], _plot_coher_spaghetti,
         r'$\mathbf{(c)}$ Coherence squared of $[\overline{u}]_s$ and $[\overline{m}]_s$'),
        (axes[1, 1], phasediff_plot_func,
         r'$\mathbf{(d)}$ Phase difference of $[\overline{u}]_s$ and $[\overline{m}]_s$'),
    ]
    for ax, plot_func, panel_title in panels:
        style_handles, ylabel, ylim, xlim, xlabel = plot_func(ax)
        _decorate_spaghetti_axis(ax, ylabel, ylim, style_handles, xlim, xlabel)
        ax.set_title(panel_title, fontsize=12)

    fig.subplots_adjust(left=0.05, right=0.87, bottom=0.06, top=0.96, hspace=0.3, wspace=0.25)
    model_legend = fig.legend(
        handles=legend_handles, loc='center left', bbox_to_anchor=(0.875, 0.5),
        fontsize=10, ncol=1, title='CMIP6 models', title_fontsize=11, frameon=False
    )

    out_file = os.path.join(plot_dir, out_name)
    fig.savefig(out_file, bbox_extra_artists=(model_legend,), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


_make_combined_figure(
    'CMIP6_fig1_cospec_coher_pdiff_spaghetti_combined.png',
    model_legend_handles, _plot_phasediff_spaghetti,
)
_make_combined_figure(
    'CMIP6_fig1a_cospec_coher_pdiff_spaghetti_combined.png',
    model_legend_handles_no_tau, _plot_phasediff_spaghetti,
)
