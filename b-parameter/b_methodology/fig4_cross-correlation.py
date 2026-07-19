"""
Reproduces the lagged cross-correlation plot (originally produced by
functions.SIT_functions.SIT_eddy_plotting_functions.eof_plots), using the
already-calculated PC1 time series, comparing the "full" (native) and "va"
variants for the same reanalysis period.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/EOF_full_100_850_prop_nans.nc                             (PC1 time series)

Southern hemisphere, all_time, EOF1 only: lagged cross-correlation of ucomp
PC1 (winds) with div1_QG's pseudo-PC1 (eddy momentum-flux divergence,
projected onto ucomp's own EOF1, since div1_QG has no independent EOF).

Produces two separate figures:
    fig4_cross-correlation_full.png : native variant
    fig4_cross-correlation_va.png   : "va" variant

Matching the plots in:
    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/level_full_100_850/
        EOF_plots/cross_correlation/s_hemisphere/all_time/ucomp/
        ucomp_div1_QG_PCs_from_ucomp_s_all_time_lagged_prop_nans.pdf
        _va/ucomp_div1_QG_va_PCs_from_ucomp_va_s_all_time_lagged_prop_nans.pdf
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
lag_len = 40  # matches lag_len used to generate the original cross-correlation plots

ucomp_name_full = f'ucomp_PCs_{hemisphere}_{time_frame}'
div1_name_full = f'div1_QG_PCs_from_ucomp_{hemisphere}_{time_frame}'

ucomp_name_va = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_name_va = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'


def compute_cross_corr(ucomp_name, div1_name):
    x_corr = eof_ds[ucomp_name].sel(eof_num=0).values
    y_corr = eof_ds[div1_name].sel(eof_num=0).values

    # pos_lags: ucomp leads div1 ; neg_lags: div1 leads ucomp
    cross_corr_pos = sm.ccf(y_corr, x_corr, nlags=lag_len)
    cross_corr_neg = sm.ccf(x_corr, y_corr, nlags=lag_len)
    return cross_corr_pos, cross_corr_neg


def plot_cross_corr(cross_corr_pos, cross_corr_neg, pos_lags, neg_lags, out_file):
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
    ax.grid(True)
    ax.set_title(r'Lagged cross-correlation of $[\overline{u}]_s$ and $[\overline{m}]_s$')

    ax.text(15, 0.35, r'$[\overline{u}]_s$ leads $[\overline{m}]_s$', fontsize=11, ha='center', va='center')
    ax.text(-15, 0.35, r'$[\overline{m}]_s$ leads $[\overline{u}]_s$', fontsize=11, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)
    print(f'Saved figure to {out_file}')


pos_lags = np.arange(lag_len)
neg_lags = np.arange(0, -lag_len, -1)

cross_corr_pos_full, cross_corr_neg_full = compute_cross_corr(ucomp_name_full, div1_name_full)
cross_corr_pos_va, cross_corr_neg_va = compute_cross_corr(ucomp_name_va, div1_name_va)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

np.savez(
    os.path.join(data_dir, 'cross_correlation_jra55.npz'),
    pos_lags=pos_lags,
    neg_lags=neg_lags,
    cross_corr_pos_full=cross_corr_pos_full,
    cross_corr_neg_full=cross_corr_neg_full,
    cross_corr_pos_va=cross_corr_pos_va,
    cross_corr_neg_va=cross_corr_neg_va,
)
print(f"Saved cross-correlation data to {data_dir}/cross_correlation_jra55.npz")

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

plot_cross_corr(
    cross_corr_pos_full, cross_corr_neg_full, pos_lags, neg_lags,
    os.path.join(plot_dir, 'fig4_cross-correlation_full.png')
)
plot_cross_corr(
    cross_corr_pos_va, cross_corr_neg_va, pos_lags, neg_lags,
    os.path.join(plot_dir, 'fig4_cross-correlation_va.png')
)
