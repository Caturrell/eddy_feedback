"""
Combines lagged-autocorrelation, "m regressed onto PC", and "feedback
strength b" panels from the Simpson et al. (2013)-style figure into a 2x2
grid: div1_QG-family variables are colour-coded, and the two level
configurations ("250,500,850hPa" 3-level subset vs "full 100-850hPa"
pressure-weighted column) are distinguished by solid vs dashed linestyle.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/EOF_250_500_850hPa_prop_nans.nc                  (PC1 timeseries)
        6hourly/EOF_full_100_850_prop_nans.nc                    (PC1 timeseries)
        6hourly/level_250_500_850hPa/b_dataset.nc                (b parameter, lag 7-14)
        6hourly/level_full_100_850/b_dataset.nc                  (b parameter, lag 7-14)

Southern hemisphere, all_time, "va" variant only.
    (a) top-left:     lagged autocorrelation of ucomp's PC1 ($[\\overline{u}]_s$)
    (b) top-right:    lagged autocorrelation of the div1_QG-family pseudo-PC1s
                       ($[\\overline{m}]_s$), one line per wavenumber band
    (c) bottom-left:  raw lag-regression slope of the div1_QG-family eddy
                       forcing pseudo-PC onto ucomp's PC1 (lag -30 to 30 days)
    (d) bottom-right: the b parameter (ratio of the slope in (c) to ucomp's
                       own lag-autoregression slope, lag 7-14 days, as
                       already computed by
                       functions.SIT_functions.SIT_eddy_feedback_functions.b_fit_simpson_2013).
                       Each line is annotated with its mean over lag 7-14,
                       placed at the right-hand wall of the panel (250,500,850hPa
                       above its line, 100-850hPa below its line, both at alpha=0.7).

NOTE ON UNITS: unlike a reference figure that regresses a physical eddy
forcing field, in m/s/day, onto a PC, here both the eddy forcing and ucomp
timeseries are pseudo-PCs derived from a unit-variance-normalised EOF1, so
the panel (c) slope is a normalised-PC regression coefficient (per day), not
literally m/s/day.
"""

import os
import numpy as np
import xarray as xar
import scipy.stats
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

level_dirs = {
    '250_500_850hPa': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'level_250_500_850hPa'
    ),
    'full_100_850': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'level_full_100_850'
    ),
}

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
va_str = '_va'
max_lag = 30

vars_to_analyse = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
var_colors = {'div1_QG': 'tab:blue', 'div1_QG_123': 'tab:orange', 'div1_QG_gt3': 'tab:green'}
var_labels = {'div1_QG': 'all k', 'div1_QG_123': 'k1-3', 'div1_QG_gt3': 'k>3'}
u_color = 'black'

level_linestyles = {'250_500_850hPa': '-', 'full_100_850': '--'}
level_labels = {'250_500_850hPa': '250,500,850hPa', 'full_100_850': '100-850hPa'}


def get_pc_timeseries(eof_ds, var_to_analyse):
    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'
    div1_name = f'{var_to_analyse}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'

    x = eof_ds[ucomp_name].sel(eof_num=0).values
    y = eof_ds[div1_name].sel(eof_num=0).values
    return x, y


def compute_m_regression(x, y, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    slopes = np.zeros(lags.shape) + np.nan

    for i, lag in enumerate(lags):
        if lag >= 0:
            x_data = x[:len(x) - lag] if lag > 0 else x
            y_data = y[lag:]
        else:
            x_data = x[-lag:]
            y_data = y[:len(y) + lag]

        mask = np.isfinite(x_data) & np.isfinite(y_data)
        slopes[i] = scipy.stats.linregress(x_data[mask], y_data[mask]).slope

    return lags, slopes


def compute_autocorr(pc1, max_lag):
    acf_vals = sm.ccf(pc1, pc1, nlags=max_lag)
    pos_lags = np.arange(max_lag)
    neg_lags = np.arange(0, -max_lag, -1)
    return pos_lags, neg_lags, acf_vals


def extract_b(b_ds, var_to_analyse):
    name = f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'
    return b_ds['lag'].values, b_ds[name].values


eof_ds_dict = {level_key: xar.open_dataset(f) for level_key, f in eof_files.items()}
b_ds_dict = {level_key: xar.open_dataset(os.path.join(d, 'b_dataset.nc')) for level_key, d in level_dirs.items()}

# ── Compute ──────────────────────────────────────────────────────────────────

u_autocorr_results = {}
m_autocorr_results = {}
m_results = {}
b_results = {}
for level_key in level_dirs:
    x_ref, _ = get_pc_timeseries(eof_ds_dict[level_key], vars_to_analyse[0])
    u_autocorr_results[level_key] = compute_autocorr(x_ref, max_lag)

    for var_to_analyse in vars_to_analyse:
        x, y = get_pc_timeseries(eof_ds_dict[level_key], var_to_analyse)

        m_autocorr_results[(var_to_analyse, level_key)] = compute_autocorr(y, max_lag)

        m_lags, m_slope = compute_m_regression(x, y, max_lag)
        m_results[(var_to_analyse, level_key)] = (m_lags, m_slope)

        b_lag, b_val = extract_b(b_ds_dict[level_key], var_to_analyse)
        b_results[(var_to_analyse, level_key)] = (b_lag, b_val)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {}
for level_key in level_dirs:
    pos_lags, neg_lags, acf_vals = u_autocorr_results[level_key]
    save_dict[f'u_autocorr_pos_lags_{level_key}'] = pos_lags
    save_dict[f'u_autocorr_neg_lags_{level_key}'] = neg_lags
    save_dict[f'u_autocorr_{level_key}'] = acf_vals

    for var_to_analyse in vars_to_analyse:
        pos_lags, neg_lags, acf_vals = m_autocorr_results[(var_to_analyse, level_key)]
        save_dict[f'm_autocorr_pos_lags_{var_to_analyse}_{level_key}'] = pos_lags
        save_dict[f'm_autocorr_neg_lags_{var_to_analyse}_{level_key}'] = neg_lags
        save_dict[f'm_autocorr_{var_to_analyse}_{level_key}'] = acf_vals

        m_lags, m_slope = m_results[(var_to_analyse, level_key)]
        b_lag, b_val = b_results[(var_to_analyse, level_key)]
        save_dict[f'm_lags_{var_to_analyse}_{level_key}'] = m_lags
        save_dict[f'm_slope_{var_to_analyse}_{level_key}'] = m_slope
        save_dict[f'b_lag_{var_to_analyse}_{level_key}'] = b_lag
        save_dict[f'b_val_{var_to_analyse}_{level_key}'] = b_val

np.savez(os.path.join(data_dir, 'm-regression_b-parameter_jra55.npz'), **save_dict)
print(f"Saved data to {data_dir}/m-regression_b-parameter_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

fig, ((ax_u_ac, ax_m_ac), (ax_m, ax_b)) = plt.subplots(2, 2, figsize=(12, 10))

for level_key in level_dirs:
    linestyle = level_linestyles[level_key]

    pos_lags, neg_lags, acf_vals = u_autocorr_results[level_key]
    ax_u_ac.plot(pos_lags, acf_vals, color=u_color, linestyle=linestyle)
    ax_u_ac.plot(neg_lags, acf_vals, color=u_color, linestyle=linestyle)

for var_to_analyse in vars_to_analyse:
    color = var_colors[var_to_analyse]
    for level_key in level_dirs:
        linestyle = level_linestyles[level_key]

        pos_lags, neg_lags, acf_vals = m_autocorr_results[(var_to_analyse, level_key)]
        ax_m_ac.plot(pos_lags, acf_vals, color=color, linestyle=linestyle)
        ax_m_ac.plot(neg_lags, acf_vals, color=color, linestyle=linestyle)

        m_lags, m_slope = m_results[(var_to_analyse, level_key)]
        ax_m.plot(m_lags, m_slope, color=color, linestyle=linestyle)

        b_lag, b_val = b_results[(var_to_analyse, level_key)]
        ax_b.plot(b_lag, b_val, color=color, linestyle=linestyle)

        mean_b = np.nanmean(b_val)
        anchor_lag = b_lag[-1]
        y_anchor = b_val[-1]

        if level_key == '250_500_850hPa':
            y_text, text_alpha = y_anchor + 0.007, 1.0
        else:
            y_text, text_alpha = y_anchor - 0.007, 0.7
        if var_to_analyse in ('div1_QG', 'div1_QG_123'):
            y_text -= 0.003

        ax_b.text(anchor_lag - 0.3, y_text, f'{mean_b:.3f}', color=color, alpha=text_alpha,
                  fontsize=8, ha='right', va='center')

for ax, title in ((ax_u_ac, r'Lagged autocorrelation of $[\overline{u}]_s$'),
                   (ax_m_ac, r'Lagged autocorrelation of $[\overline{m}]_s$')):
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.2, 1.0)
    ax.set_yticks(np.arange(-0.2, 1.0 + 0.2, 0.2))
    ax.grid(True)
    ax.set_xlabel('lag (days)')
    ax.set_ylabel('lagged correlation')
    ax.set_title(title)

ax_m.axhline(0, color='k', linewidth=0.5)
ax_m.set_xlim(-30, 30)
ax_m.grid(True)
ax_m.set_xlabel('lag (days)')
ax_m.set_ylabel(r'$[\overline{m}]_s$ (normalised PC coeff./day)')
ax_m.set_title(r'$[\overline{m}]_s$ regressed onto PC (vertically integrated)')

ax_b.axhline(0, color='k', linewidth=0.5)
ax_b.set_xlim(7, 14)
ax_b.set_ylim(-0.1, 0.1)
ax_b.grid(True)
ax_b.set_xlabel('lag (days)')
ax_b.set_ylabel(r'$b$')
ax_b.set_title('Feedback strength, b (SH; vertically integrated)')

panel_labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip((ax_u_ac, ax_m_ac, ax_m, ax_b), panel_labels):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

color_handles = [Line2D([0], [0], color=var_colors[v], linestyle='-', label=var_labels[v]) for v in vars_to_analyse]
linestyle_handles = [Line2D([0], [0], color='k', linestyle=level_linestyles[k], label=level_labels[k])
                     for k in level_dirs]

color_legend = fig.legend(handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=9, frameon=False)
fig.add_artist(color_legend)
fig.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 1])

out_file = os.path.join(plot_dir, 'fig5_m-regression_b-parameter.png')
plt.savefig(out_file, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure to {out_file}')
