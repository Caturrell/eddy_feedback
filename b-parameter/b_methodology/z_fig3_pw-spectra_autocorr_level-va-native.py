"""
Variant of fig3_pw-spectra_autocorr.py: compares 4 versions of the
power-spectrum/lagged-autocorrelation figure at once -- the two level
configurations ("250,500,850hPa" 3-level subset vs "full 100-850hPa"
pressure-weighted column) crossed with the "va" and native (non-va) variants
of ucomp/div1_QG.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/level_250_500_850hPa/power_spec.nc          (power spectra)
        6hourly/level_full_100_850/power_spec.nc            (power spectra)
        6hourly/EOF_250_500_850hPa_prop_nans.nc              (PC1 time series)
        6hourly/EOF_full_100_850_prop_nans.nc                (PC1 time series)

Southern hemisphere, all_time, div1_QG (all wavenumbers), EOF1 only:
    (a) top-left:     power spectrum of ucomp PC1     (winds)
    (b) top-right:    lagged autocorrelation of ucomp PC1          (winds)
    (c) bottom-left:  power spectrum of div1_QG PC1   (eddy momentum-flux divergence)
    (d) bottom-right: lagged autocorrelation of div1_QG PC1        (eddy momentum-flux divergence)

Colour denotes level configuration (blue: 250,500,850hPa; orange: full
100-850hPa); linestyle denotes variant (solid: va; dashed: native), matching
z_fig4_cross-correlation_level-va-native.py's colour/linestyle convention.

Matching the plots in:
    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/
        level_250_500_850hPa/power_spec_plots/s/all_time/_va/
        level_full_100_850/power_spec_plots/s/all_time/_va/
        EOF_plots/autocorrelation/s_hemisphere/all_time/_va/
"""

import os
import numpy as np
import xarray as xar
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

power_spec_files = {
    '250_500_850hPa': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'level_250_500_850hPa', 'power_spec.nc'
    ),
    'full_100_850': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'level_full_100_850', 'power_spec.nc'
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
time_name = 'time'
frequency_name = f'frequency_{time_name}'
lag_len = 40  # matches lag_len used to generate the original autocorrelation plots

variant_va_str = {'va': '_va', 'native': ''}
variant_linestyles = {'va': '-', 'native': '--'}
variant_labels = {'va': 'vertically-averaged', 'native': 'full-field'}

level_colors = {'250_500_850hPa': 'tab:blue', 'full_100_850': 'tab:orange'}
level_labels = {'250_500_850hPa': '250,500,850hPa', 'full_100_850': '100-850hPa'}

power_spec_ds_dict = {level_key: xar.open_dataset(f, auto_complex=True) for level_key, f in power_spec_files.items()}
eof_ds_dict = {level_key: xar.open_dataset(f) for level_key, f in eof_files.items()}

freq = power_spec_ds_dict['full_100_850'][frequency_name]
pos_lags = np.arange(lag_len)
neg_lags = np.arange(0, -lag_len, -1)


def compute_power_spectra(power_spec_ds, va_str):
    # Factor of 2 matches functions.SIT_functions.SIT_eddy_plotting_functions.plot_power_spectrum
    ucomp_ps_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'
    div1_ps_name = f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'

    ucomp_power_spec = 2. * power_spec_ds[f'{ucomp_ps_name}_power_spec_stft'].values
    div1_power_spec = 2. * power_spec_ds[f'{div1_ps_name}_power_spec_stft'].values
    return ucomp_power_spec, div1_power_spec


def compute_autocorr(eof_ds, va_str):
    ucomp_autocorr_name = f'ucomp{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'
    div1_autocorr_name = f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'

    ucomp_pc1 = eof_ds[ucomp_autocorr_name].sel(eof_num=0).values
    div1_pc1 = eof_ds[div1_autocorr_name].sel(eof_num=0).values
    ucomp_acf = sm.ccf(ucomp_pc1, ucomp_pc1, nlags=lag_len)
    div1_acf = sm.ccf(div1_pc1, div1_pc1, nlags=lag_len)
    return ucomp_acf, div1_acf


# ── Compute ──────────────────────────────────────────────────────────────────

power_spec_results = {}
autocorr_results = {}
for level_key in power_spec_ds_dict:
    for variant_key, va_str in variant_va_str.items():
        power_spec_results[(level_key, variant_key)] = compute_power_spectra(power_spec_ds_dict[level_key], va_str)
        autocorr_results[(level_key, variant_key)] = compute_autocorr(eof_ds_dict[level_key], va_str)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {'freq': freq.values, 'pos_lags': pos_lags, 'neg_lags': neg_lags}
for (level_key, variant_key), (ucomp_power_spec, div1_power_spec) in power_spec_results.items():
    save_dict[f'ucomp_power_spec_{level_key}_{variant_key}'] = ucomp_power_spec
    save_dict[f'div1_power_spec_{level_key}_{variant_key}'] = div1_power_spec
for (level_key, variant_key), (ucomp_acf, div1_acf) in autocorr_results.items():
    save_dict[f'ucomp_acf_{level_key}_{variant_key}'] = ucomp_acf
    save_dict[f'div1_acf_{level_key}_{variant_key}'] = div1_acf

np.savez(os.path.join(data_dir, 'pw_spectra_autocorr_level-va-native_jra55.npz'), **save_dict)
print(f"Saved power-spectra/autocorrelation data to {data_dir}/pw_spectra_autocorr_level-va-native_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex='col')

for level_key in power_spec_ds_dict:
    color = level_colors[level_key]
    for variant_key in variant_va_str:
        linestyle = variant_linestyles[variant_key]

        ucomp_power_spec, div1_power_spec = power_spec_results[(level_key, variant_key)]
        axes[0, 0].plot(freq, ucomp_power_spec, color=color, linestyle=linestyle)
        axes[1, 0].plot(freq, div1_power_spec, color=color, linestyle=linestyle)

        ucomp_acf, div1_acf = autocorr_results[(level_key, variant_key)]
        axes[0, 1].plot(pos_lags, ucomp_acf, color=color, linestyle=linestyle)
        axes[0, 1].plot(neg_lags, ucomp_acf, color=color, linestyle=linestyle)
        axes[1, 1].plot(pos_lags, div1_acf, color=color, linestyle=linestyle)
        axes[1, 1].plot(neg_lags, div1_acf, color=color, linestyle=linestyle)

for ax, title in ((axes[0, 0], r'Power spectrum of $[\overline{u}]_s$'),
                   (axes[1, 0], r'Power spectrum of $[\overline{m}]_s$')):
    ax.set_xlim(0., 0.25)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('frequency (1/days)')
    ax.set_ylabel('power spectral density')

for ax, title in ((axes[0, 1], r'Lagged autocorrelation of $[\overline{u}]_s$'),
                   (axes[1, 1], r'Lagged autocorrelation of $[\overline{m}]_s$')):
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-30., 30.)
    ax.set_ylim(-0.2, 1.0)
    ax.set_yticks(np.arange(-0.2, 1.0 + 0.2, 0.2))
    ax.set_xlabel('lag (days)')
    ax.set_ylabel('lagged correlation')
    ax.grid(True)
    ax.set_title(title)

panel_labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip(axes.flat, panel_labels):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

color_handles = [Line2D([0], [0], color=level_colors[k], linestyle='-', label=level_labels[k]) for k in power_spec_ds_dict]
linestyle_handles = [Line2D([0], [0], color='k', linestyle=variant_linestyles[k], label=variant_labels[k])
                     for k in variant_va_str]

color_legend = fig.legend(handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=9, frameon=False)
fig.add_artist(color_legend)
fig.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 1])

plot_dir = os.path.join(script_dir, 'plots', 'z_extra_analysis')
os.makedirs(plot_dir, exist_ok=True)

out_file = os.path.join(plot_dir, 'z_fig3_pw-spectra_autocorr_level-va-native.png')
plt.savefig(out_file, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure to {out_file}')
