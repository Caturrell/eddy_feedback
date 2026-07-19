"""
Collates the lagged cross-correlation diagnostic (originally produced by
functions.SIT_functions.SIT_eddy_plotting_functions.eof_plots) across all
CMIP6 models into a grid figure (one panel per model) and a spaghetti figure
(all models overlaid on one axes, plus the JRA55 reanalysis as a thick black
reference line).

Source data (PC1 time series): /gws/ssde/j25a/arctic_connect/cturrell/CMIP6/
    historical/<model>/<time_span>/6hrPlevPt/1979_2015/EOF_prop_nans.nc
    (time_span is whichever of 1850_2015/1850_2014/1950_2015/1950_2014 exists
    for that model - see all_plots_true/hist_calc_efp_b.py)
JRA55 reference data: b-parameter/b_methodology/data/cross_correlation_jra55.npz

Southern hemisphere, all_time, "va" variant, EOF1 only: lagged cross-correlation
of ucomp PC1 (winds) with div1_QG's pseudo-PC1 (eddy momentum-flux divergence,
projected onto ucomp's own EOF1, since div1_QG has no independent EOF).

Matching the plot in:
    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/level_full_100_850/
        EOF_plots/cross_correlation/s_hemisphere/all_time/ucomp/_va/
        ucomp_div1_QG_va_PCs_from_ucomp_va_s_all_time_lagged_prop_nans.pdf
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
lag_len = 40  # matches lag_len used to generate the original cross-correlation plot

ucomp_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'

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
    eof_file = _find_eof_file(model)
    if eof_file is None:
        skipped_models.append(model)
        continue

    try:
        with xar.open_dataset(eof_file) as eof_ds:
            x_corr = eof_ds[ucomp_name].sel(eof_num=0).values
            y_corr = eof_ds[div1_name].sel(eof_num=0).values

        # pos_lags: ucomp leads div1 ; neg_lags: div1 leads ucomp
        models_data[model] = {
            'cross_corr_pos': sm.ccf(y_corr, x_corr, nlags=lag_len),
            'cross_corr_neg': sm.ccf(x_corr, y_corr, nlags=lag_len),
        }
    except KeyError:
        skipped_models.append(model)

if skipped_models:
    print(f'Warning: skipped {len(skipped_models)} model(s) with no usable '
          f'EOF_prop_nans.nc: {", ".join(skipped_models)}')

used_models = sorted(models_data)
print(f'Plotting {len(used_models)} model(s): {", ".join(used_models)}')

jra55_file = os.path.normpath(
    os.path.join(script_dir, '..', 'data', 'cross_correlation_jra55.npz')
)
jra55_data = np.load(jra55_file)

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# The per-model grid goes in a subdir; the spaghetti figure (this quantity's
# collated summary) is saved directly under plot_dir.
subplot_dir = os.path.join(plot_dir, 'fig4_cross_correlation')
os.makedirs(subplot_dir, exist_ok=True)

n_models = len(used_models)
n_cols = 6
n_rows = int(np.ceil(n_models / n_cols))

model_colors = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, n_models))

# ---------------------------------------------------------------------------
# Grid figure (one panel per model)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(3. * n_cols, 2.5 * n_rows), sharex=True
)
axes = np.atleast_2d(axes)
fig_title = fig.suptitle(
    r'Lagged cross-correlation of $[\overline{u}]_s$ and $[\overline{m}]_s$ (all CMIP6 models)',
    fontsize=14
)
for extra_ax in axes.flat[n_models:]:
    extra_ax.axis('off')

legend_handles = None
for ax, model in zip(axes.flat, used_models):
    data = models_data[model]
    (h,) = ax.plot(pos_lags, data['cross_corr_pos'], color='blue', label='cross-corr')
    ax.plot(neg_lags, data['cross_corr_neg'], color='blue')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-30., 30.)
    ax.set_ylim(-0.2, 0.6)
    ax.set_yticks(np.arange(-0.2, 0.6 + 0.2, 0.2))
    ax.grid(True, axis='y')
    ax.tick_params(labelsize=7)
    ax.set_title(model, fontsize=8)
    if legend_handles is None:
        legend_handles = [h]

for row in axes:
    row[0].set_ylabel('lagged correlation', fontsize=8)
active_ids = {id(ax) for ax in axes.flat[:n_models]}
for col in range(n_cols):
    col_active = [ax for ax in axes[:, col] if id(ax) in active_ids]
    if col_active:
        col_active[-1].set_xlabel('lag (days)', fontsize=8)

fig.legend(handles=legend_handles, loc='lower center', ncol=1, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02 / n_rows))
fig.tight_layout(rect=(0., 0.02, 1., 0.97))

grid_out_file = os.path.join(subplot_dir, 'CMIP6_fig4_cross-correlation_grid.png')
fig.savefig(grid_out_file, bbox_extra_artists=(fig_title,), bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved {grid_out_file}')

# ---------------------------------------------------------------------------
# Spaghetti figure (all models overlaid, JRA55 as thick black reference)
# ---------------------------------------------------------------------------
jra55_legend_handle = plt.Line2D([0], [0], color='k', lw=2.5, label='JRA55')

model_legend_handles = [
    plt.Line2D([0], [0], color=color, lw=1.2, label=model)
    for model, color in zip(used_models, model_colors)
] + [jra55_legend_handle]

fig, ax = plt.subplots(figsize=(15., 6.5))
ax.set_title(
    'Lagged cross-correlation of $[\\overline{u}]_s$ and $[\\overline{m}]_s$',
    fontsize=14
)

for model, color in zip(used_models, model_colors):
    data = models_data[model]
    ax.plot(pos_lags, data['cross_corr_pos'], color=color, lw=0.8, alpha=0.7)
    ax.plot(neg_lags, data['cross_corr_neg'], color=color, lw=0.8, alpha=0.7)
ax.axhline(0, color='0.3', linewidth=0.5)
ax.axvline(0, color='0.3', linewidth=0.5)
ax.plot(pos_lags, jra55_data['cross_corr_pos'], color='k', lw=2.5)
ax.plot(neg_lags, jra55_data['cross_corr_neg'], color='k', lw=2.5)

ax.set_xlim(-30., 30.)
ax.set_ylim(-0.2, 0.6)
ax.set_yticks(np.arange(-0.2, 0.6 + 0.2, 0.2))
ax.set_xlabel('lag (days)', fontsize=11)
ax.set_ylabel('lagged correlation', fontsize=11)
ax.tick_params(labelsize=11)
ax.grid(True, axis='y')

ax.text(15, 0.55, r'$[\overline{u}]_s$ leads $[\overline{m}]_s$', fontsize=11, ha='center', va='center')
ax.text(-15, 0.55, r'$[\overline{m}]_s$ leads $[\overline{u}]_s$', fontsize=11, ha='center', va='center')

fig.subplots_adjust(right=0.72, top=0.9)

model_legend = ax.legend(
    handles=model_legend_handles, loc='center left', bbox_to_anchor=(1.005, 0.5),
    fontsize=10, ncol=1, title='CMIP6 models', title_fontsize=11, frameon=False
)

out_file = os.path.join(plot_dir, 'CMIP6_fig4_cross-correlation_spaghetti.png')
fig.savefig(out_file, bbox_extra_artists=(model_legend,),
            bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved {out_file}')
