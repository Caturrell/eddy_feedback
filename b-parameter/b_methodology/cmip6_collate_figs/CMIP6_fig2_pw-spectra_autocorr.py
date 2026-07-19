"""
Collates the power-spectrum and lagged-autocorrelation diagnostics (originally
four separate PDFs produced by functions.SIT_functions.SIT_eddy_plotting_functions)
across all CMIP6 models into grid figures (one panel per model) and spaghetti
figures (all models overlaid on one axes, plus the JRA55 reanalysis as a thick
black reference line).

Source data:
    power spectra: b-parameter/b_methodology/all_plots_true/250-500-850hPa_dm/
        1979_2015/<model>/6hrPlevPt/power_spec.nc
    PC1 time series (for autocorrelation): /gws/ssde/j25a/arctic_connect/cturrell/
        CMIP6/historical/<model>/<time_span>/6hrPlevPt/1979_2015/EOF_prop_nans.nc
        (time_span is whichever of 1850_2015/1850_2014/1950_2015/1950_2014 exists
        for that model - see all_plots_true/hist_calc_efp_b.py)
JRA55 reference data: b-parameter/b_methodology/data/pw_spectra_autocorr_jra55.npz

Uses div1_QG (southern hemisphere, all_time, "va" variant, EOF1 only), matching
the per-model plots in
all_plots_true/250-500-850hPa_dm/1979_2015/<model>/6hrPlevPt/
power_spec_plots/s/all_time/_va/ and
EOF_plots/autocorrelation/s_hemisphere/all_time/_va/
"""

import os
import numpy as np
import xarray as xar
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

cmip6_base_dir = os.path.normpath(os.path.join(
    script_dir, '..', 'all_plots_true', '250-500-850hPa_dm', '1979_2015'
))
eof_base_dir = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
possible_time_spans = ['1850_2015', '1850_2014', '1950_2015', '1950_2014']

hemisphere = 's'
time_frame = 'all_time'
time_name = 'time'
frequency_name = f'frequency_{time_name}'
lag_len = 40  # matches lag_len used to generate the original autocorrelation plots

ucomp_ps_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_ps_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'
ucomp_autocorr_name = f'ucomp_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'
div1_autocorr_name = div1_ps_name

pos_lags = np.arange(lag_len)
neg_lags = np.arange(0, -lag_len, -1)

model_names = sorted(
    d for d in os.listdir(cmip6_base_dir)
    if os.path.isdir(os.path.join(cmip6_base_dir, d))
)


def _find_eof_file(model):
    for time_span in possible_time_spans:
        candidate = os.path.join(
            eof_base_dir, model, time_span, '6hrPlevPt', '1979_2015', 'EOF_prop_nans.nc'
        )
        if os.path.isfile(candidate):
            return candidate
    return None


models_data = {}
skipped_models = []

for model in model_names:
    power_spec_file = os.path.join(cmip6_base_dir, model, '6hrPlevPt', 'power_spec.nc')
    eof_file = _find_eof_file(model)

    if not os.path.isfile(power_spec_file) or eof_file is None:
        skipped_models.append(model)
        continue

    try:
        with xar.open_dataset(power_spec_file, auto_complex=True) as power_spec_ds:
            freq = power_spec_ds[frequency_name].values
            ucomp_power_spec = 2. * power_spec_ds[f'{ucomp_ps_name}_power_spec_stft'].values
            div1_power_spec = 2. * power_spec_ds[f'{div1_ps_name}_power_spec_stft'].values

        with xar.open_dataset(eof_file) as eof_ds:
            ucomp_pc1 = eof_ds[ucomp_autocorr_name].sel(eof_num=0).values
            div1_pc1 = eof_ds[div1_autocorr_name].sel(eof_num=0).values

        models_data[model] = {
            'freq': freq,
            'ucomp_power_spec': ucomp_power_spec,
            'div1_power_spec': div1_power_spec,
            'ucomp_acf': sm.ccf(ucomp_pc1, ucomp_pc1, nlags=lag_len),
            'div1_acf': sm.ccf(div1_pc1, div1_pc1, nlags=lag_len),
        }
    except KeyError:
        skipped_models.append(model)

if skipped_models:
    print(f'Warning: skipped {len(skipped_models)} model(s) with no usable '
          f'power_spec.nc / EOF_prop_nans.nc: {", ".join(skipped_models)}')

used_models = sorted(models_data)
print(f'Plotting {len(used_models)} model(s): {", ".join(used_models)}')

jra55_file = os.path.normpath(
    os.path.join(script_dir, '..', 'data', 'pw_spectra_autocorr_jra55.npz')
)
jra55_data = np.load(jra55_file)

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Individual grid/spaghetti figures go in a subdir; only the combined
# spaghetti figure is saved directly under plot_dir.
subplot_dir = os.path.join(plot_dir, 'fig2_pw_spectra_autocorr')
os.makedirs(subplot_dir, exist_ok=True)

n_models = len(used_models)
n_cols = 6
n_rows = int(np.ceil(n_models / n_cols))

model_colors = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, n_models))


def _make_grid_figure(suptitle):
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3. * n_cols, 2.5 * n_rows), sharex=True
    )
    axes = np.atleast_2d(axes)
    fig.suptitle(suptitle, fontsize=14)
    for extra_ax in axes.flat[n_models:]:
        extra_ax.axis('off')
    return fig, axes


def _finish_and_save(fig, axes, out_name, ylabel, legend_handles,
                      xlim=(0., 0.25), xlabel='frequency (1/days)'):
    for ax in axes.flat[:n_models]:
        ax.set_xlim(*xlim)
        ax.grid(True)
        ax.tick_params(labelsize=7)
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=8)
    # bottom-most active axis in each column gets the x-label
    active_ids = {id(ax) for ax in axes.flat[:n_models]}
    for col in range(n_cols):
        col_active = [ax for ax in axes[:, col] if id(ax) in active_ids]
        if col_active:
            col_active[-1].set_xlabel(xlabel, fontsize=8)

    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02 / n_rows))
    fig.tight_layout(rect=(0., 0.02, 1., 0.97))

    out_file = os.path.join(subplot_dir, out_name)
    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out_file}')


# ---------------------------------------------------------------------------
# Power-spectrum grids
# ---------------------------------------------------------------------------
def _power_spec_grid(quantity_key, suptitle, out_name):
    fig, axes = _make_grid_figure(suptitle)
    legend_handles = None
    for ax, model in zip(axes.flat, used_models):
        data = models_data[model]
        (h,) = ax.plot(data['freq'], data[quantity_key], color='blue', label='stft')
        ax.set_title(model, fontsize=8)
        if legend_handles is None:
            legend_handles = [h]
    _finish_and_save(fig, axes, out_name, 'power spectral density', legend_handles)


_power_spec_grid('ucomp_power_spec',
                  r'Power spectrum of $[\overline{u}]_s$ (all CMIP6 models)',
                  'CMIP6_fig2_ucomp-power-spec_grid.png')
_power_spec_grid('div1_power_spec',
                  r'Power spectrum of $[\overline{m}]_s$ (all CMIP6 models)',
                  'CMIP6_fig2_div1-power-spec_grid.png')

# ---------------------------------------------------------------------------
# Lagged-autocorrelation grids
# ---------------------------------------------------------------------------
def _autocorr_grid(quantity_key, suptitle, out_name):
    fig, axes = _make_grid_figure(suptitle)
    legend_handles = None
    for ax, model in zip(axes.flat, used_models):
        acf_vals = models_data[model][quantity_key]
        (h,) = ax.plot(pos_lags, acf_vals, color='blue', label='ACF')
        ax.plot(neg_lags, acf_vals, color='blue')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_ylim(-0.2, 1.0)
        ax.set_yticks(np.arange(-0.2, 1.0 + 0.2, 0.2))
        ax.set_title(model, fontsize=8)
        if legend_handles is None:
            legend_handles = [h]
    _finish_and_save(fig, axes, out_name, 'lagged correlation', legend_handles,
                      xlim=(-30., 30.), xlabel='lag (days)')


_autocorr_grid('ucomp_acf',
                r'Lagged autocorrelation of $[\overline{u}]_s$ (all CMIP6 models)',
                'CMIP6_fig2_ucomp-autocorr_grid.png')
_autocorr_grid('div1_acf',
                r'Lagged autocorrelation of $[\overline{m}]_s$ (all CMIP6 models)',
                'CMIP6_fig2_div1-autocorr_grid.png')


# ---------------------------------------------------------------------------
# Spaghetti plots (all models overlaid, JRA55 as thick black reference)
# ---------------------------------------------------------------------------
jra55_legend_handle = plt.Line2D([0], [0], color='k', lw=2.5, label='JRA55')

model_legend_handles = [
    plt.Line2D([0], [0], color=color, lw=1.2, label=model)
    for model, color in zip(used_models, model_colors)
] + [jra55_legend_handle]


def _plot_ucomp_powerspec_spaghetti(ax):
    for model, color in zip(used_models, model_colors):
        data = models_data[model]
        ax.plot(data['freq'], data['ucomp_power_spec'], color=color, lw=0.8, alpha=0.7)
    ax.plot(jra55_data['freq'], jra55_data['ucomp_power_spec'], color='k', lw=2.5)
    return [], 'power spectral density', (0., 90.), (0., 0.25), 'frequency (1/days)'


def _plot_div1_powerspec_spaghetti(ax):
    for model, color in zip(used_models, model_colors):
        data = models_data[model]
        ax.plot(data['freq'], data['div1_power_spec'], color=color, lw=0.8, alpha=0.7)
    ax.plot(jra55_data['freq'], jra55_data['div1_power_spec'], color='k', lw=2.5)
    return [], 'power spectral density', (0., 2.4), (0., 0.25), 'frequency (1/days)'


def _plot_ucomp_autocorr_spaghetti(ax):
    for model, color in zip(used_models, model_colors):
        acf_vals = models_data[model]['ucomp_acf']
        ax.plot(pos_lags, acf_vals, color=color, lw=0.8, alpha=0.7)
        ax.plot(neg_lags, acf_vals, color=color, lw=0.8, alpha=0.7)
    ax.axhline(0, color='0.3', linewidth=0.5)
    ax.axvline(0, color='0.3', linewidth=0.5)
    ax.plot(pos_lags, jra55_data['ucomp_acf'], color='k', lw=2.5)
    ax.plot(neg_lags, jra55_data['ucomp_acf'], color='k', lw=2.5)
    return [], 'lagged correlation', (-0.2, 1.0), (-30., 30.), 'lag (days)'


def _plot_div1_autocorr_spaghetti(ax):
    for model, color in zip(used_models, model_colors):
        acf_vals = models_data[model]['div1_acf']
        ax.plot(pos_lags, acf_vals, color=color, lw=0.8, alpha=0.7)
        ax.plot(neg_lags, acf_vals, color=color, lw=0.8, alpha=0.7)
    ax.axhline(0, color='0.3', linewidth=0.5)
    ax.axvline(0, color='0.3', linewidth=0.5)
    ax.plot(pos_lags, jra55_data['div1_acf'], color='k', lw=2.5)
    ax.plot(neg_lags, jra55_data['div1_acf'], color='k', lw=2.5)
    return [], 'lagged correlation', (-0.2, 1.0), (-30., 30.), 'lag (days)'


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
    r'Power spectrum of $[\overline{u}]_s$ - CMIP6 models vs JRA55',
    _plot_ucomp_powerspec_spaghetti, 'CMIP6_fig2_ucomp-power-spec_spaghetti.png'
)
_make_single_spaghetti_figure(
    r'Power spectrum of $[\overline{m}]_s$ - CMIP6 models vs JRA55',
    _plot_div1_powerspec_spaghetti, 'CMIP6_fig2_div1-power-spec_spaghetti.png'
)
_make_single_spaghetti_figure(
    r'Lagged autocorrelation of $[\overline{u}]_s$ - CMIP6 models vs JRA55',
    _plot_ucomp_autocorr_spaghetti, 'CMIP6_fig2_ucomp-autocorr_spaghetti.png'
)
_make_single_spaghetti_figure(
    r'Lagged autocorrelation of $[\overline{m}]_s$ - CMIP6 models vs JRA55',
    _plot_div1_autocorr_spaghetti, 'CMIP6_fig2_div1-autocorr_spaghetti.png'
)

# ---------------------------------------------------------------------------
# Combined spaghetti plot (2x2, matching CMIP6_fig2_pw-spectra_autocorr.py layout)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16., 12.))

panels = [
    (axes[0, 0], _plot_ucomp_powerspec_spaghetti,
     r'$\mathbf{(a)}$ Power spectrum of $[\overline{u}]_s$'),
    (axes[0, 1], _plot_ucomp_autocorr_spaghetti,
     r'$\mathbf{(b)}$ Lagged autocorrelation of $[\overline{u}]_s$'),
    (axes[1, 0], _plot_div1_powerspec_spaghetti,
     r'$\mathbf{(c)}$ Power spectrum of $[\overline{m}]_s$'),
    (axes[1, 1], _plot_div1_autocorr_spaghetti,
     r'$\mathbf{(d)}$ Lagged autocorrelation of $[\overline{m}]_s$'),
]
for ax, plot_func, panel_title in panels:
    style_handles, ylabel, ylim, xlim, xlabel = plot_func(ax)
    _decorate_spaghetti_axis(ax, ylabel, ylim, style_handles, xlim, xlabel)
    ax.set_title(panel_title, fontsize=12)

fig.subplots_adjust(left=0.05, right=0.87, bottom=0.06, top=0.96, hspace=0.3, wspace=0.25)
model_legend = fig.legend(
    handles=model_legend_handles, loc='center left', bbox_to_anchor=(0.875, 0.5),
    fontsize=10, ncol=1, title='CMIP6 models', title_fontsize=11, frameon=False
)

out_file = os.path.join(plot_dir, 'CMIP6_fig2_pw-spectra_autocorr_spaghetti_combined.png')
fig.savefig(out_file, bbox_extra_artists=(model_legend,), bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved {out_file}')
