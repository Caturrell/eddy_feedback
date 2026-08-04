"""
Combines z_fig2_cospec_coher_pdiff_level-va-native.py and
z_fig3_pw-spectra_autocorr_level-va-native.py into a single 4-row, 2-column
figure. Each original 2x2 figure is kept as-is (full titles, default sizing,
its own axis-sharing/labelling behaviour) and simply stacked vertically:
fig2 occupies rows 1-2, fig3 occupies rows 3-4.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/level_250_500_850hPa/power_spec.nc
        6hourly/level_full_100_850/power_spec.nc
        6hourly/EOF_250_500_850hPa_prop_nans.nc
        6hourly/EOF_full_100_850_prop_nans.nc

Southern hemisphere, all_time, div1_QG (all wavenumbers), EOF1 only:
    Row 1: (a) Re(cospectrum), (b) Im(cospectrum) of [u]_s and [m]_s
    Row 2: (c) coherence^2,    (d) phase difference of [u]_s and [m]_s
    Row 3: (e) power spectrum of [u]_s,   (f) lagged autocorrelation of [u]_s
    Row 4: (g) power spectrum of [m]_s,   (h) lagged autocorrelation of [m]_s

Colour denotes level configuration (blue: 250,500,850hPa; orange: full
100-850hPa); linestyle denotes variant (solid: va; dashed: native).

Rows 3-4 replicate fig3's own column-wise x-axis sharing (col 0: frequency,
col 1: lag), scoped to just those two rows -- row 3 keeps its axis-label text
but its tick numbers are hidden since they'd duplicate row 4's.
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
lag_len = 40  # matches lag_len used to generate the original autocorrelation/cross-corr plots

variant_va_str = {'va': '_va', 'native': ''}
variant_linestyles = {'va': '-', 'native': '--'}
variant_labels = {'va': 'vertically-averaged', 'native': 'non-averaged'}

level_colors = {'250_500_850hPa': 'tab:blue', 'full_100_850': 'tab:orange'}
level_labels = {'250_500_850hPa': '250,500,850hPa', 'full_100_850': '100-850hPa'}

power_spec_ds_dict = {level_key: xar.open_dataset(f, auto_complex=True) for level_key, f in power_spec_files.items()}
eof_ds_dict = {level_key: xar.open_dataset(f) for level_key, f in eof_files.items()}

freq = power_spec_ds_dict['full_100_850'][frequency_name]
pos_lags = np.arange(lag_len)
neg_lags = np.arange(0, -lag_len, -1)


def extract_cospec_fields(power_spec_ds, va_str):
    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'
    div1_name = f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'

    cospec = power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft']
    coher = power_spec_ds[f'{ucomp_name}_{div1_name}_coher_stft']
    phase_diff = power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff']
    tau_fit_3 = float(power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'])
    return cospec.values, coher.values, phase_diff.values, tau_fit_3


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

cospec_results = {}
power_spec_results = {}
autocorr_results = {}
for level_key in power_spec_ds_dict:
    for variant_key, va_str in variant_va_str.items():
        cospec_results[(level_key, variant_key)] = extract_cospec_fields(power_spec_ds_dict[level_key], va_str)
        power_spec_results[(level_key, variant_key)] = compute_power_spectra(power_spec_ds_dict[level_key], va_str)
        autocorr_results[(level_key, variant_key)] = compute_autocorr(eof_ds_dict[level_key], va_str)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {'freq': freq.values, 'pos_lags': pos_lags, 'neg_lags': neg_lags}
for (level_key, variant_key), (cospec, coher, phase_diff, tau_fit_3) in cospec_results.items():
    save_dict[f'cospec_{level_key}_{variant_key}'] = cospec
    save_dict[f'coher_{level_key}_{variant_key}'] = coher
    save_dict[f'phase_diff_{level_key}_{variant_key}'] = phase_diff
    save_dict[f'tau_fit_3_{level_key}_{variant_key}'] = np.array(tau_fit_3)
for (level_key, variant_key), (ucomp_power_spec, div1_power_spec) in power_spec_results.items():
    save_dict[f'ucomp_power_spec_{level_key}_{variant_key}'] = ucomp_power_spec
    save_dict[f'div1_power_spec_{level_key}_{variant_key}'] = div1_power_spec
for (level_key, variant_key), (ucomp_acf, div1_acf) in autocorr_results.items():
    save_dict[f'ucomp_acf_{level_key}_{variant_key}'] = ucomp_acf
    save_dict[f'div1_acf_{level_key}_{variant_key}'] = div1_acf

np.savez(os.path.join(data_dir, 'fig4a_compare_all_va_native_jra55.npz'), **save_dict)
print(f"Saved combined data to {data_dir}/fig4a_compare_all_va_native_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 2, figsize=(13, 18))
ax_re, ax_im = axes[0]
ax_coher, ax_pdiff = axes[1]
ax_ups, ax_uac = axes[2]
ax_dps, ax_dac = axes[3]

for level_key in power_spec_ds_dict:
    color = level_colors[level_key]
    for variant_key in variant_va_str:
        linestyle = variant_linestyles[variant_key]

        cospec, coher, phase_diff, _ = cospec_results[(level_key, variant_key)]
        ax_re.plot(freq, np.real(cospec), color=color, linestyle=linestyle)
        ax_im.plot(freq, np.imag(cospec), color=color, linestyle=linestyle)
        ax_coher.plot(freq, coher ** 2., color=color, linestyle=linestyle)
        ax_pdiff.plot(freq, phase_diff, color=color, linestyle=linestyle)

        ucomp_power_spec, div1_power_spec = power_spec_results[(level_key, variant_key)]
        ax_ups.plot(freq, ucomp_power_spec, color=color, linestyle=linestyle)
        ax_dps.plot(freq, div1_power_spec, color=color, linestyle=linestyle)

        ucomp_acf, div1_acf = autocorr_results[(level_key, variant_key)]
        ax_uac.plot(pos_lags, ucomp_acf, color=color, linestyle=linestyle)
        ax_uac.plot(neg_lags, ucomp_acf, color=color, linestyle=linestyle)
        ax_dac.plot(pos_lags, div1_acf, color=color, linestyle=linestyle)
        ax_dac.plot(neg_lags, div1_acf, color=color, linestyle=linestyle)

# ── Row 1-2: fig2, as-is ─────────────────────────────────────────────────────

ax_re.plot(freq, 2. * np.pi * freq, color='k', linestyle=':', label=r'$\omega = 2\pi f$')
ax_re.legend()
ax_re.set_xlim(0., 0.25)
ax_re.set_ylim(0., 2.5)
ax_re.set_yticks(np.arange(0., 2.5 + 0.5, 0.5))
ax_re.grid(True)
ax_re.set_title(r'$\mathbf{Real\ part}$: cospectrum of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax_re.set_xlabel('frequency (1/days)')

ax_im.plot(freq, 2. * np.pi * freq, color='k', linestyle=':', label=r'$\omega = 2\pi f$')
ax_im.legend()
ax_im.set_xlim(0., 0.25)
ax_im.set_ylim(0., 2.5)
ax_im.set_yticks(np.arange(0., 2.5 + 0.5, 0.5))
ax_im.grid(True)
ax_im.set_title(r'$\mathbf{Imag\ part}$: cospectrum of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax_im.set_xlabel('frequency (1/days)')

ax_coher.set_xlim(0., 0.25)
ax_coher.set_ylim(0., 1.)
ax_coher.set_yticks(np.arange(0., 1. + 0.2, 0.2))
ax_coher.grid(True)
ax_coher.set_title(r'Coherence squared of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax_coher.set_xlabel('frequency (1/days)')

ax_pdiff.set_xlim(0., 0.25)
ax_pdiff.set_ylim(0., 90.)
ax_pdiff.set_yticks(np.arange(0., 90. + 22.5, 22.5))
ax_pdiff.grid(True)
ax_pdiff.set_title(r'Phase difference of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax_pdiff.set_xlabel('frequency (1/days)')

# ── Row 3-4: fig3, as-is (column-wise x-sharing scoped to these two rows) ───

ax_ups.sharex(ax_dps)
ax_uac.sharex(ax_dac)
ax_ups.tick_params(labelbottom=False)
ax_uac.tick_params(labelbottom=False)

for ax, title in ((ax_ups, r'Power spectrum of $[\overline{u}]_s$'),
                   (ax_dps, r'Power spectrum of $[\overline{m}]_s$')):
    ax.set_xlim(0., 0.25)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('frequency (1/days)')
    ax.set_ylabel('power spectral density')

for ax, title in ((ax_uac, r'Lagged autocorrelation of $[\overline{u}]_s$'),
                   (ax_dac, r'Lagged autocorrelation of $[\overline{m}]_s$')):
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-30., 30.)
    ax.set_ylim(-0.2, 1.0)
    ax.set_yticks(np.arange(-0.2, 1.0 + 0.2, 0.2))
    ax.set_xlabel('lag (days)')
    ax.set_ylabel('lagged correlation')
    ax.grid(True)
    ax.set_title(title)

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
for ax, label in zip(axes.flat, panel_labels):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

color_handles = [Line2D([0], [0], color=level_colors[k], linestyle='-', label=level_labels[k]) for k in power_spec_ds_dict]
linestyle_handles = [Line2D([0], [0], color='k', linestyle=variant_linestyles[k], label=variant_labels[k])
                     for k in variant_va_str]

color_legend = fig.legend(handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=9, frameon=False)
fig.add_artist(color_legend)
fig.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.035, 1, 1])

plot_dir = os.path.join(script_dir, 'plots', 'z_extra_analysis')
os.makedirs(plot_dir, exist_ok=True)

out_file = os.path.join(plot_dir, 'z_fig4a_compare_all_va_native.png')
plt.savefig(out_file, bbox_inches='tight', dpi=150)
plt.close(fig)

print(f'Saved combined figure to {out_file}')
