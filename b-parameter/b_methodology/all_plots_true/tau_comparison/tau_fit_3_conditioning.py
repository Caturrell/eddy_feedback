"""
Read-only diagnostic extending tau_comparison.py: tests whether the size of the
disagreement between tau_fit_3 (original, cospectral) and tau_autocorr_efold
(Simpson sliding, autocorrelation e-fold) tracks how well-conditioned tau_fit_3's
own low-frequency regression is, for every hemisphere x time_frame x variant
combo - not just the SH DJF case originally diagnosed.

Background: tau_fit_3 = slope(Im cospectrum) / intercept(Re cospectrum), a linear
regression over the 7 points with freq <= 0.025 day^-1 (see
compute_phase_difference, functions/SIT_functions/SIT_eddy_feedback_functions.py:151-192).
SH DJF va's 14.5-day outlier was traced to a small Re-intercept (0.068, vs
0.12-0.22 elsewhere) amplifying regression noise into the slope/intercept ratio.
This script recomputes that same 7-point regression (real_intercept,
real_intercept_stderr, real_rsq, imag_rsq) plus the actual number of STFT
segments underlying the spectral estimate, for every combo, and checks by
Spearman correlation whether intercept conditioning (SNR or R^2) actually
predicts |Delta_tau| in general, or whether the pattern seen at SH DJF is
something more specific to that one point.

Does not modify tau_fit_3, tau_autocorr_efold, or b_fit_simpson_2013 - reads
only the already-computed power_spec.nc (jra55_850) and tau_simpson_sliding.nc
(jra_simpson_lag), same sources as tau_comparison.py, same script/data/plots
directory layout.
"""

import os

import numpy as np
import pandas as pd
import scipy.stats
import xarray as xar
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
all_plots_true_dir = os.path.dirname(script_dir)

SEASONS = [
    'all_time',
    'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
    'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ',
]
HEMISPHERES = ['n', 's']
va_str_dict = {'native': '', 'va': '_va'}
LOW_FREQ_CUTOFF = 0.025  # matches compute_phase_difference's where_low_freq

old_power_spec_file = os.path.join(
    all_plots_true_dir, 'jra55_850_sit_plots', '1979_2014', '6hourly',
    'level_250_500_850hPa', 'power_spec.nc')
new_tau_file = os.path.join(
    all_plots_true_dir, 'jra_simpson_lag_sit_plots', '1979_2014', '6hourly',
    'level_250_500_850hPa', 'tau_simpson_sliding.nc')

old_ds = xar.open_dataset(old_power_spec_file, auto_complex=True)
new_ds = xar.open_dataset(new_tau_file)


def var_names(va_str, hemisphere, season):
    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{season}'
    div1_name = f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{season}'
    freq_name = f'frequency_time_{season}' if season != 'all_time' else 'frequency_time'
    return ucomp_name, div1_name, freq_name


def diagnostics(va_str, hemisphere, season):
    """Recomputes the exact 7-point low-frequency regression compute_phase_difference
    uses for tau_fit_3, to expose its conditioning (intercept, stderr, R^2), plus the
    actual number of STFT segments underlying the spectral estimate for this combo."""
    ucomp_name, div1_name, freq_name = var_names(va_str, hemisphere, season)

    freq = old_ds[freq_name].values
    omega = 2. * np.pi * freq
    cospec = old_ds[f'{ucomp_name}_{div1_name}_cospec_stft'].values
    low = np.where(freq <= LOW_FREQ_CUTOFF)[0]

    real = np.real(cospec)[low]
    imag = np.imag(cospec)[low]

    lr_real = scipy.stats.linregress(omega[low], real)
    lr_imag = scipy.stats.linregress(omega[low], imag)

    tau_fit_3 = float(old_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'].values)
    n_segments = old_ds[f'{ucomp_name}_fourier_coeffs_stft'].shape[1]

    return {
        'n_low_freq_points': len(low),
        'n_segments': int(n_segments),
        'real_intercept': lr_real.intercept,
        'real_intercept_stderr': lr_real.intercept_stderr,
        'real_rsq': lr_real.rvalue ** 2,
        'imag_rsq': lr_imag.rvalue ** 2,
        'intercept_snr': abs(lr_real.intercept) / lr_real.intercept_stderr,
        'tau_fit_3': tau_fit_3,
    }


# ── Build the combined diagnostic table ───────────────────────────────────────

rows = []
for hemisphere in HEMISPHERES:
    for season in SEASONS:
        for variant, va_str in va_str_dict.items():
            diag = diagnostics(va_str, hemisphere, season)
            tau_autocorr_efold = float(new_ds[f'ucomp{va_str}_tau_autocorr_efold_{hemisphere}_{season}'].values)
            delta_tau = tau_autocorr_efold - diag['tau_fit_3']

            rows.append({
                'dataset': 'jra55',
                'hemisphere': hemisphere,
                'season': season,
                'variant': variant,
                'tau_fit_3': diag['tau_fit_3'],
                'tau_autocorr_efold': tau_autocorr_efold,
                'delta_tau': delta_tau,
                'abs_delta_tau': abs(delta_tau),
                'n_low_freq_points': diag['n_low_freq_points'],
                'n_segments': diag['n_segments'],
                'real_intercept': diag['real_intercept'],
                'real_intercept_stderr': diag['real_intercept_stderr'],
                'real_rsq': diag['real_rsq'],
                'imag_rsq': diag['imag_rsq'],
                'intercept_snr': diag['intercept_snr'],
            })

conditioning_df = pd.DataFrame(rows)

data_dir = os.path.join(script_dir, 'data')
plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

csv_file = os.path.join(data_dir, 'tau_fit_3_conditioning.csv')
conditioning_df.to_csv(csv_file, index=False)
print(f'Saved data to {csv_file}')

print('\nSegment counts by season (should be ~104 for all_time, fewer and season-length-dependent otherwise):')
print(conditioning_df.groupby('season')['n_segments'].agg(['min', 'max']).loc[SEASONS])

# ── Test the conditioning hypothesis: does low intercept_SNR / R^2 predict ────
# large |Delta_tau|? Try both candidate metrics, report both, and use whichever
# separates the data more clearly (larger |Spearman rho|) for the plots below.

rho_snr, p_snr = scipy.stats.spearmanr(conditioning_df['intercept_snr'], conditioning_df['abs_delta_tau'])
rho_rsq, p_rsq = scipy.stats.spearmanr(conditioning_df['real_rsq'], conditioning_df['abs_delta_tau'])

print(f"\nSpearman(intercept_SNR, |Delta_tau|) = {rho_snr:.3f} (p={p_snr:.3f})")
print(f"Spearman(real_R^2,      |Delta_tau|) = {rho_rsq:.3f} (p={p_rsq:.3f})")

if abs(rho_rsq) >= abs(rho_snr):
    metric_col, metric_label, rho, p_value = 'real_rsq', 'Re(cospectrum) fit R²', rho_rsq, p_rsq
else:
    metric_col, metric_label, rho, p_value = 'intercept_snr', 'intercept SNR (|intercept| / stderr)', rho_snr, p_snr

print(f'\nUsing {metric_col} for the plots below (larger |Spearman rho|).')
print('Note the sign: a positive rho here means BETTER conditioning goes with LARGER')
print('|Delta_tau|, i.e. the opposite of the naive "noisy fit -> big disagreement"')
print('hypothesis - worth checking the sign printed above before interpreting the plot.')


# ── Plot 1: scatter, |Delta_tau| vs conditioning metric ───────────────────────

hemisphere_color = {'n': '#1f77b4', 's': '#d62728'}
variant_marker = {'native': 'o', 'va': '^'}

fig, ax = plt.subplots(figsize=(9, 7))
for hemisphere in HEMISPHERES:
    for variant in va_str_dict.keys():
        sub = conditioning_df[(conditioning_df['hemisphere'] == hemisphere) & (conditioning_df['variant'] == variant)]
        ax.scatter(sub[metric_col], sub['abs_delta_tau'], color=hemisphere_color[hemisphere],
                   marker=variant_marker[variant], s=50, edgecolor='black', linewidth=0.4,
                   label=f'{hemisphere}H ({variant})')

known_point = conditioning_df[(conditioning_df['hemisphere'] == 's') & (conditioning_df['season'] == 'DJF') & (conditioning_df['variant'] == 'va')].iloc[0]
ax.annotate('SH DJF (va)\n(originally diagnosed outlier)',
            xy=(known_point[metric_col], known_point['abs_delta_tau']),
            xytext=(20, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

ax.set_xlabel(metric_label)
ax.set_ylabel('|Delta_tau| = |tau_autocorr_efold - tau_fit_3| (days)')
ax.set_title('Does tau_fit_3 regression conditioning predict its disagreement with tau_autocorr_efold?')
ax.grid(True)
ax.legend(fontsize=8)

ax.text(0.03, 0.97, f'Spearman rho = {rho:.3f} (p = {p_value:.3f})\nn = {len(conditioning_df)}',
        transform=ax.transAxes, fontsize=10, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
scatter_file = os.path.join(plot_dir, 'tau_fit_3_conditioning_scatter.png')
plt.savefig(scatter_file)
plt.close(fig)
print(f'Saved figure to {scatter_file}')


# ── Plot 2: bar chart, styled to match tau_comparison_{hemisphere}hemisphere.png ──

variant_hatch = {'native': '', 'va': '//'}

for hemisphere in HEMISPHERES:
    sub = conditioning_df[conditioning_df['hemisphere'] == hemisphere]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(SEASONS))
    bar_width = 0.35

    for variant, offset in [('native', -0.5 * bar_width), ('va', 0.5 * bar_width)]:
        variant_rows = sub[sub['variant'] == variant].set_index('season').loc[SEASONS]
        ax.bar(x + offset, variant_rows[metric_col].values, width=bar_width,
               color='#9467bd', hatch=variant_hatch[variant], edgecolor='black',
               linewidth=0.5, label=f'{variant}')

    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel(metric_label)
    ax.set_xlabel('season')
    ax.set_title(f'{hemisphere}H: tau_fit_3 regression conditioning ({metric_label}) - '
                 'solid=native, hatched=va')
    ax.grid(True, axis='y')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(plot_dir, f'tau_fit_3_conditioning_{hemisphere}hemisphere.png')
    plt.savefig(out_file)
    plt.close(fig)
    print(f'Saved figure to {out_file}')

print('Done.')
