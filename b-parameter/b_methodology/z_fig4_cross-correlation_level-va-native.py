"""
Variant of fig4_cross-correlation.py: compares 4 lagged cross-correlations
at once -- the two level configurations ("250,500,850hPa" 3-level subset vs
"full 100-850hPa" pressure-weighted column) crossed with the "va" and native
(non-va) variants of ucomp/div1_QG.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/EOF_250_500_850hPa_prop_nans.nc                  (PC1 time series)
        6hourly/EOF_full_100_850_prop_nans.nc                    (PC1 time series)

Southern hemisphere, all_time, EOF1 only: lagged cross-correlation of ucomp
PC1 (winds) with div1_QG's pseudo-PC1 (eddy momentum-flux divergence,
projected onto ucomp's own EOF1, since div1_QG has no independent EOF).

Colour denotes level configuration (blue: 250,500,850hPa; orange: full
100-850hPa); linestyle denotes variant (solid: va; dashed: native), matching
z_fig5_250-500-850_va-vs-native.py's va/native convention.

Matching the plots in:
    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/
        level_250_500_850hPa/EOF_plots/cross_correlation/s_hemisphere/all_time/ucomp/
            _va/ucomp_div1_QG_va_PCs_from_ucomp_va_s_all_time_lagged_prop_nans.pdf
            ucomp_div1_QG_PCs_from_ucomp_s_all_time_lagged_prop_nans.pdf
        level_full_100_850/EOF_plots/cross_correlation/s_hemisphere/all_time/ucomp/
            _va/ucomp_div1_QG_va_PCs_from_ucomp_va_s_all_time_lagged_prop_nans.pdf
            ucomp_div1_QG_PCs_from_ucomp_s_all_time_lagged_prop_nans.pdf
"""

import os
import numpy as np
import xarray as xar
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

eof_files = {
    '250_500_850hPa': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'EOF_250_500_850hPa_prop_nans.nc'
    ),
    'full_100_850': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'EOF_full_100_850_prop_nans.nc'
    ),
}

hemisphere = 's'
time_frame = 'all_time'
lag_len = 40  # matches lag_len used to generate the original cross-correlation plots

variant_va_str = {'va': '_va', 'native': ''}
variant_linestyles = {'va': '-', 'native': '--'}
variant_labels = {'va': 'vertically-averaged', 'native': 'full-field'}

level_colors = {'250_500_850hPa': 'tab:blue', 'full_100_850': 'tab:orange'}
level_labels = {'250_500_850hPa': '250,500,850hPa', 'full_100_850': '100-850hPa'}

eof_ds_dict = {level_key: xar.open_dataset(f) for level_key, f in eof_files.items()}


def compute_cross_corr(eof_ds, va_str):
    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'
    div1_name = f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'

    x_corr = eof_ds[ucomp_name].sel(eof_num=0).values
    y_corr = eof_ds[div1_name].sel(eof_num=0).values

    # pos_lags: ucomp leads div1 ; neg_lags: div1 leads ucomp
    cross_corr_pos = sm.ccf(y_corr, x_corr, nlags=lag_len)
    cross_corr_neg = sm.ccf(x_corr, y_corr, nlags=lag_len)
    return cross_corr_pos, cross_corr_neg


pos_lags = np.arange(lag_len)
neg_lags = np.arange(0, -lag_len, -1)

# ── Compute ──────────────────────────────────────────────────────────────────

results = {}
for level_key, eof_ds in eof_ds_dict.items():
    for variant_key, va_str in variant_va_str.items():
        cross_corr_pos, cross_corr_neg = compute_cross_corr(eof_ds, va_str)
        results[(level_key, variant_key)] = (cross_corr_pos, cross_corr_neg)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {'pos_lags': pos_lags, 'neg_lags': neg_lags}
for (level_key, variant_key), (cross_corr_pos, cross_corr_neg) in results.items():
    save_dict[f'cross_corr_pos_{level_key}_{variant_key}'] = cross_corr_pos
    save_dict[f'cross_corr_neg_{level_key}_{variant_key}'] = cross_corr_neg

np.savez(os.path.join(data_dir, 'cross_correlation_level-va-native_jra55.npz'), **save_dict)
print(f"Saved cross-correlation data to {data_dir}/cross_correlation_level-va-native_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

plot_dir = os.path.join(script_dir, 'plots', 'z_extra_analysis')
os.makedirs(plot_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))

for level_key in eof_ds_dict:
    color = level_colors[level_key]
    for variant_key in variant_va_str:
        linestyle = variant_linestyles[variant_key]
        cross_corr_pos, cross_corr_neg = results[(level_key, variant_key)]

        ax.plot(pos_lags, cross_corr_pos, color=color, linestyle=linestyle)
        ax.plot(neg_lags, cross_corr_neg, color=color, linestyle=linestyle)

ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlim(-30., 30.)
ax.set_ylim(-0.2, 0.6)
ax.set_yticks(np.arange(-0.2, 0.6 + 0.2, 0.2))

ax.set_xlabel('lag (days)')
ax.set_ylabel('lagged correlation')
ax.grid(True)
ax.set_title(r'Lagged cross-correlation of $[\overline{u}]_s$ and $[\overline{m}]_s$')

ax.text(15, 0.35, r'$[\overline{u}]_s$ leads $[\overline{m}]_s$', fontsize=11, ha='center', va='center')
ax.text(-15, 0.35, r'$[\overline{m}]_s$ leads $[\overline{u}]_s$', fontsize=11, ha='center', va='center')

color_handles = [Line2D([0], [0], color=level_colors[k], linestyle='-', label=level_labels[k]) for k in eof_ds_dict]
linestyle_handles = [Line2D([0], [0], color='k', linestyle=variant_linestyles[k], label=variant_labels[k])
                     for k in variant_va_str]

color_legend = fig.legend(handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=9, frameon=False)
fig.add_artist(color_legend)
fig.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])

out_file = os.path.join(plot_dir, 'z_fig4_cross-correlation_level-va-native.png')
plt.savefig(out_file, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure to {out_file}')
