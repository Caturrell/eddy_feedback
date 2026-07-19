"""
Variant of fig2_cospec_coher_pdiff.py: compares 4 versions of the
cospectrum/coherence/phase-difference figure at once -- the two level
configurations ("250,500,850hPa" 3-level subset vs "full 100-850hPa"
pressure-weighted column) crossed with the "va" and native (non-va) variants
of ucomp/div1_QG.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/level_250_500_850hPa/power_spec.nc
        6hourly/level_full_100_850/power_spec.nc

Southern hemisphere, all_time, div1_QG (all wavenumbers) only.

Colour denotes level configuration (blue: 250,500,850hPa; orange: full
100-850hPa); linestyle denotes variant (solid: va; dashed: native), matching
z_fig4_cross-correlation_level-va-native.py's colour/linestyle convention.

The cospectrum's real and imaginary parts are plotted as two separate panels
(rather than overlaid in one, which would need a third visual channel on top
of the colour/linestyle already used for level/variant). The
phase-difference arctan fit curves are dropped here to keep the 4-line
panels legible, though the underlying tau_fit_3 values are still saved.

Matching the plots in:
    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/
        level_250_500_850hPa/power_spec_plots/s/all_time/_va/
        level_full_100_850/power_spec_plots/s/all_time/_va/
"""

import os
import numpy as np
import xarray as xar
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

hemisphere = 's'
time_frame = 'all_time'
time_name = 'time'
frequency_name = f'frequency_{time_name}'

variant_va_str = {'va': '_va', 'native': ''}
variant_linestyles = {'va': '-', 'native': '--'}
variant_labels = {'va': 'vertically-averaged', 'native': 'full-field'}

level_colors = {'250_500_850hPa': 'tab:blue', 'full_100_850': 'tab:orange'}
level_labels = {'250_500_850hPa': '250,500,850hPa', 'full_100_850': '100-850hPa'}

power_spec_ds_dict = {level_key: xar.open_dataset(f, auto_complex=True) for level_key, f in power_spec_files.items()}

freq = power_spec_ds_dict['full_100_850'][frequency_name]


def extract_fields(power_spec_ds, va_str):
    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'
    div1_name = f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'

    cospec = power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft']
    coher = power_spec_ds[f'{ucomp_name}_{div1_name}_coher_stft']
    phase_diff = power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff']
    tau_fit_3 = float(power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'])
    return cospec.values, coher.values, phase_diff.values, tau_fit_3


# ── Compute ──────────────────────────────────────────────────────────────────

results = {}
for level_key, power_spec_ds in power_spec_ds_dict.items():
    for variant_key, va_str in variant_va_str.items():
        results[(level_key, variant_key)] = extract_fields(power_spec_ds, va_str)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {'freq': freq.values}
for (level_key, variant_key), (cospec, coher, phase_diff, tau_fit_3) in results.items():
    save_dict[f'cospec_{level_key}_{variant_key}'] = cospec
    save_dict[f'coher_{level_key}_{variant_key}'] = coher
    save_dict[f'phase_diff_{level_key}_{variant_key}'] = phase_diff
    save_dict[f'tau_fit_3_{level_key}_{variant_key}'] = np.array(tau_fit_3)

np.savez(os.path.join(data_dir, 'cospec_coher_pdiff_level-va-native_jra55.npz'), **save_dict)
print(f"Saved cospectrum/coherence/phase-difference data to "
      f"{data_dir}/cospec_coher_pdiff_level-va-native_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
ax_re, ax_im, ax_coher, ax_pdiff = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

for level_key in power_spec_ds_dict:
    color = level_colors[level_key]
    for variant_key in variant_va_str:
        linestyle = variant_linestyles[variant_key]
        cospec, _, _, _ = results[(level_key, variant_key)]
        ax_re.plot(freq, np.real(cospec), color=color, linestyle=linestyle)
        ax_im.plot(freq, np.imag(cospec), color=color, linestyle=linestyle)
ax_re.plot(freq, 2. * np.pi * freq, color='k', linestyle=':', label=r'$\omega = 2\pi f$')
ax_re.legend()
ax_re.set_xlim(0., 0.25)
ax_re.set_ylim(0., 1.75)
ax_re.set_yticks(np.arange(0., 1.75 + 0.25, 0.25))
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

for level_key in power_spec_ds_dict:
    color = level_colors[level_key]
    for variant_key in variant_va_str:
        linestyle = variant_linestyles[variant_key]
        _, coher, _, _ = results[(level_key, variant_key)]
        ax_coher.plot(freq, coher ** 2., color=color, linestyle=linestyle)
ax_coher.set_xlim(0., 0.25)
ax_coher.set_ylim(0., 1.)
ax_coher.set_yticks(np.arange(0., 1. + 0.2, 0.2))
ax_coher.grid(True)
ax_coher.set_title(r'Coherence squared of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax_coher.set_xlabel('frequency (1/days)')

for level_key in power_spec_ds_dict:
    color = level_colors[level_key]
    for variant_key in variant_va_str:
        linestyle = variant_linestyles[variant_key]
        _, _, phase_diff, _ = results[(level_key, variant_key)]
        ax_pdiff.plot(freq, phase_diff, color=color, linestyle=linestyle)
ax_pdiff.set_xlim(0., 0.25)
ax_pdiff.set_ylim(0., 90.)
ax_pdiff.set_yticks(np.arange(0., 90. + 22.5, 22.5))
ax_pdiff.grid(True)
ax_pdiff.set_title(r'Phase difference of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax_pdiff.set_xlabel('frequency (1/days)')

panel_labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip((ax_re, ax_im, ax_coher, ax_pdiff), panel_labels):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

color_handles = [Line2D([0], [0], color=level_colors[k], linestyle='-', label=level_labels[k]) for k in power_spec_ds_dict]
linestyle_handles = [Line2D([0], [0], color='k', linestyle=variant_linestyles[k], label=variant_labels[k])
                     for k in variant_va_str]

color_legend = fig.legend(handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=9, frameon=False)
fig.add_artist(color_legend)
fig.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])

plot_dir = os.path.join(script_dir, 'plots', 'z_extra_analysis')
os.makedirs(plot_dir, exist_ok=True)

out_file = os.path.join(plot_dir, 'z_fig2_cospec_coher_pdiff_level-va-native.png')
plt.savefig(out_file, bbox_inches='tight')
plt.close()

print(f'Saved combined figure to {out_file}')
