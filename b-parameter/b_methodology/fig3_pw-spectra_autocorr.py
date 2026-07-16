"""
Combines the power-spectrum and lagged-autocorrelation plots (originally four
separate PDFs produced by functions.SIT_functions.SIT_eddy_plotting_functions)
into a single 2x2 figure, using the already-calculated data.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/level_full_100_850/power_spec.nc                          (power spectra)
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/EOF_full_100_850_prop_nans.nc                             (PC1 time series)

Layout (southern hemisphere, all_time, "va" variant, EOF1 only):
    (a) top-left:     power spectrum of ucomp PC1     (winds)
    (b) top-right:    lagged autocorrelation of ucomp PC1          (winds)
    (c) bottom-left:  power spectrum of div1_QG PC1   (eddy momentum-flux divergence)
    (d) bottom-right: lagged autocorrelation of div1_QG PC1        (eddy momentum-flux divergence)

Matching the plots in:
    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/level_full_100_850/
        power_spec_plots/s/all_time/_va/
        EOF_plots/autocorrelation/s_hemisphere/all_time/_va/
"""

import os
import numpy as np
import xarray as xar
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

power_spec_file = os.path.join(
    script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
    '6hourly', 'level_full_100_850', 'power_spec.nc'
)
eof_file = os.path.join(
    script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
    '6hourly', 'EOF_full_100_850_prop_nans.nc'
)

power_spec_ds = xar.open_dataset(power_spec_file, auto_complex=True)
eof_ds = xar.open_dataset(eof_file)

hemisphere = 's'
time_frame = 'all_time'
time_name = 'time'
frequency_name = f'frequency_{time_name}'
lag_len = 40  # matches lag_len used to generate the original autocorrelation plots

freq = power_spec_ds[frequency_name]

# Power spectra use the "native" ucomp PC1, and div1_QG's pseudo-PC1
# (projected onto ucomp's own EOF1, since div1_QG has no independent EOF).
ucomp_ps_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_ps_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'

# Autocorrelation uses the pseudo-PC1 for both variables (matching the
# requested source PDFs) -- numerically identical to the native ucomp PC1.
ucomp_autocorr_name = f'ucomp_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'
div1_autocorr_name = div1_ps_name


def plot_power_spectrum_panel(ax, ps_name, title):
    # Factor of 2 matches functions.SIT_functions.SIT_eddy_plotting_functions.plot_power_spectrum
    ax.plot(freq, 2. * power_spec_ds[f'{ps_name}_power_spec_stft'], color='blue')
    ax.set_xlim(0., 0.25)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('frequency (1/days)')
    ax.set_ylabel('power spectral density')


def plot_autocorrelation_panel(ax, var_name, title):
    pc1 = eof_ds[var_name].sel(eof_num=0).values
    acf_vals = sm.ccf(pc1, pc1, nlags=lag_len)
    pos_lags = np.arange(lag_len)
    neg_lags = np.arange(0, -lag_len, -1)

    ax.plot(pos_lags, acf_vals, color='blue')
    ax.plot(neg_lags, acf_vals, color='blue')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-30., 30.)
    ax.set_ylim(-0.2, 1.0)
    ax.set_yticks(np.arange(-0.2, 1.0 + 0.2, 0.2))

    ax.set_xlabel('lag (days)')
    ax.set_ylabel('lagged correlation')
    ax.grid(True)
    ax.set_title(title)


# ── Save data ────────────────────────────────────────────────────────────────

ucomp_power_spec = 2. * power_spec_ds[f'{ucomp_ps_name}_power_spec_stft'].values
div1_power_spec = 2. * power_spec_ds[f'{div1_ps_name}_power_spec_stft'].values

ucomp_pc1 = eof_ds[ucomp_autocorr_name].sel(eof_num=0).values
div1_pc1 = eof_ds[div1_autocorr_name].sel(eof_num=0).values
ucomp_acf = sm.ccf(ucomp_pc1, ucomp_pc1, nlags=lag_len)
div1_acf = sm.ccf(div1_pc1, div1_pc1, nlags=lag_len)
pos_lags = np.arange(lag_len)
neg_lags = np.arange(0, -lag_len, -1)

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

np.savez(
    os.path.join(data_dir, 'pw_spectra_autocorr_jra55.npz'),
    freq=freq.values,
    ucomp_power_spec=ucomp_power_spec,
    div1_power_spec=div1_power_spec,
    pos_lags=pos_lags,
    neg_lags=neg_lags,
    ucomp_acf=ucomp_acf,
    div1_acf=div1_acf,
)
print(f"Saved power-spectra/autocorrelation data to {data_dir}/pw_spectra_autocorr_jra55.npz")

fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex='col')

plot_power_spectrum_panel(axes[0, 0], ucomp_ps_name, r'Power spectrum of $[\overline{u}]_s$')
plot_autocorrelation_panel(axes[0, 1], ucomp_autocorr_name, r'Lagged autocorrelation of $[\overline{u}]_s$')
plot_power_spectrum_panel(axes[1, 0], div1_ps_name, r'Power spectrum of $[\overline{m}]_s$')
plot_autocorrelation_panel(axes[1, 1], div1_autocorr_name, r'Lagged autocorrelation of $[\overline{m}]_s$')

panel_labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip(axes.flat, panel_labels):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

plt.tight_layout()

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

out_file = os.path.join(plot_dir, 'fig3_pw-spectra_autocorr.png')
plt.savefig(out_file)
plt.close()

print(f'Saved combined figure to {out_file}')
