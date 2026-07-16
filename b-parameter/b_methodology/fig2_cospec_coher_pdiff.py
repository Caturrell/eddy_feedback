"""
Combines the cospectrum density, coherence-squared, and phase-difference plots
(originally three separate PDFs produced by
functions.SIT_functions.SIT_eddy_feedback_functions.power_spectrum_analysis)
into a single figure, using the already-calculated power spectrum data.

Source data: b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/
             1979_2014/6hourly/level_full_100_850/power_spec.nc

Uses div1_QG (southern hemisphere, all_time, "va" variant), matching the plots in
all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/level_full_100_850/
power_spec_plots/s/all_time/_va/
"""

import os
import numpy as np
import xarray as xar
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

power_spec_file = os.path.join(
    script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
    '6hourly', 'level_full_100_850', 'power_spec.nc'
)

power_spec_ds = xar.open_dataset(power_spec_file, auto_complex=True)

hemisphere = 's'
time_frame = 'all_time'
time_name = 'time'
frequency_name = f'frequency_{time_name}'

ucomp_name = f'ucomp_va_PCs_{hemisphere}_{time_frame}'
div1_name = f'div1_QG_va_PCs_from_ucomp_va_{hemisphere}_{time_frame}'

freq = power_spec_ds[frequency_name]

cospec = power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft']
coher = power_spec_ds[f'{ucomp_name}_{div1_name}_coher_stft']
phase_diff = power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff']
tau_fit_1 = float(power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_1'])
tau_fit_2 = float(power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_2'])
tau_fit_3 = float(power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'])

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

np.savez(
    os.path.join(data_dir, 'cospec_coher_pdiff_jra55.npz'),
    freq=freq.values,
    cospec=cospec.values,
    coher=coher.values,
    phase_diff=phase_diff.values,
    tau_fit_1=np.array(tau_fit_1),
    tau_fit_2=np.array(tau_fit_2),
    tau_fit_3=np.array(tau_fit_3),
)
print(f"Saved cospectrum/coherence/phase-difference data to {data_dir}/cospec_coher_pdiff_jra55.npz")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

panel_labels = ['(a)', '(b)', '(c)']

ax = axes[0]
ax.plot(freq, np.real(cospec), label='real')
ax.plot(freq, np.imag(cospec), label='Imag')
ax.plot(freq, 2. * np.pi * freq, label=r'$2\pi f$', linestyle='--')
ax.set_xlim(0., 0.25)
ax.set_ylim(0., 1.75)
ax.set_yticks(np.arange(0., 1.75 + 0.25, 0.25))
ax.legend()
ax.grid(True)
ax.set_title(r'Cospectrum of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax.set_xlabel('frequency (1/days)')

ax = axes[1]
ax.plot(freq, coher ** 2.)
ax.set_xlim(0., 0.25)
ax.set_ylim(0., 1.)
ax.set_yticks(np.arange(0., 1. + 0.2, 0.2))
ax.grid(True)
ax.set_title(r'Coherence squared of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax.set_xlabel('frequency (1/days)')

ax = axes[2]
ax.plot(freq, phase_diff, label='observed')
# ax.plot(freq, np.rad2deg(np.arctan(2. * np.pi * freq * tau_fit_1)),
#         linestyle='--', label=f'fit with {tau_fit_1:4.2f} days')
# ax.plot(freq, np.rad2deg(np.arctan(2. * np.pi * freq * tau_fit_2)),
#         linestyle='--', label=f'fit with {tau_fit_2:4.2f} days')
ax.plot(freq, np.rad2deg(np.arctan(2. * np.pi * freq * tau_fit_3)),
        linestyle='--', color='#d62728', label=f'arctan({tau_fit_3:4.2f}*$\\omega$)')
ax.set_xlim(0., 0.25)
ax.set_ylim(0., 90.)
ax.set_yticks(np.arange(0., 90. + 22.5, 22.5))
ax.legend()
ax.grid(True)
ax.set_title(r'Phase difference of $[\overline{u}]_s$ and $[\overline{m}]_s$')
ax.set_xlabel('frequency (1/days)')

for ax, label in zip(axes, panel_labels):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

plt.tight_layout()

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

out_file = os.path.join(plot_dir, 'fig2_cospec_coher_pdiff.png')
plt.savefig(out_file)
plt.close()

print(f'Saved combined figure to {out_file}')
