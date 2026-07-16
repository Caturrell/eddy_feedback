"""
Reproduces the lagged cross-correlation plot (originally produced by
functions.SIT_functions.SIT_eddy_plotting_functions.eof_plots), using the
already-calculated PC1 time series.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/EOF_full_100_850_prop_nans.nc                             (PC1 time series)

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

eof_file = os.path.join(
    script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
    '6hourly', 'EOF_full_100_850_prop_nans.nc'
)

eof_ds = xar.open_dataset(eof_file)

hemisphere = 's'
time_frame = 'all_time'
lag_len = 40  # matches lag_len used to generate the original cross-correlation plot

ucomp_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'

x_corr = eof_ds[ucomp_name].sel(eof_num=0).values
y_corr = eof_ds[div1_name].sel(eof_num=0).values

# pos_lags: ucomp leads div1 ; neg_lags: div1 leads ucomp
cross_corr_pos = sm.ccf(y_corr, x_corr, nlags=lag_len)
cross_corr_neg = sm.ccf(x_corr, y_corr, nlags=lag_len)
pos_lags = np.arange(lag_len)
neg_lags = np.arange(0, -lag_len, -1)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

np.savez(
    os.path.join(data_dir, 'cross_correlation_jra55.npz'),
    pos_lags=pos_lags,
    neg_lags=neg_lags,
    cross_corr_pos=cross_corr_pos,
    cross_corr_neg=cross_corr_neg,
)
print(f"Saved cross-correlation data to {data_dir}/cross_correlation_jra55.npz")

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(pos_lags, cross_corr_pos, color='blue')
ax.plot(neg_lags, cross_corr_neg, color='blue')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlim(-30., 30.)
ax.set_ylim(-0.2, 0.6)
ax.set_yticks(np.arange(-0.2, 0.6 + 0.2, 0.2))

ax.set_xlabel('lag (days)')
ax.set_ylabel('lagged correlation')
ax.grid(True, axis='y')
ax.set_title(r'Lagged cross-correlation of $[\overline{u}]_s$ and $[\overline{m}]_s$')

ax.text(15, 0.55, r'$[\overline{u}]_s$ leads', fontsize=11, ha='center', va='center')
ax.text(-15, 0.55, r'$[\overline{m}]_s$ leads', fontsize=11, ha='center', va='center')

plt.tight_layout()

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

out_file = os.path.join(plot_dir, 'fig4_cross-correlation.png')
plt.savefig(out_file)
plt.close()

print(f'Saved figure to {out_file}')
