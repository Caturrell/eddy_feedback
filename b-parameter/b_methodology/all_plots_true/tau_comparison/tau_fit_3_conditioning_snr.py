"""
Re-plots the tau_fit_3 conditioning diagnostic from tau_fit_3_conditioning.py using
intercept_SNR (|intercept| / standard_error(intercept)) instead of Re(cospectrum) fit
R^2, to directly test the original SH DJF mechanism: a small Re-intercept relative to
its own regression uncertainty amplifying noise through the slope/intercept ratio into
a large tau_fit_3 error - as opposed to R^2, which tests general fit quality and (per
tau_fit_3_conditioning.py's results) does not explain the pattern seen in NH OND/NDJ.

Pure read of the already-saved data/tau_fit_3_conditioning.csv (52 rows: hemisphere x
time_frame x variant) - no recomputation, no changes to tau_fit_3, tau_autocorr_efold,
b_fit_simpson_2013, or the CSV itself.
"""

import os

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

SEASONS = [
    'all_time',
    'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
    'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ',
]
HEMISPHERES = ['n', 's']
va_str_dict = {'native': '', 'va': '_va'}

data_dir = os.path.join(script_dir, 'data')
plot_dir = os.path.join(script_dir, 'plots')

csv_file = os.path.join(data_dir, 'tau_fit_3_conditioning.csv')
conditioning_df = pd.read_csv(csv_file)
print(f'Loaded {len(conditioning_df)} rows from {csv_file}')

# Choose linear vs log x-axis based on the actual spread of intercept_snr.
snr_min, snr_max = conditioning_df['intercept_snr'].min(), conditioning_df['intercept_snr'].max()
snr_span_decades = np.log10(snr_max / snr_min)
use_log_x = snr_span_decades > 1.
print(f'intercept_snr range: {snr_min:.2f} - {snr_max:.2f} ({snr_span_decades:.2f} decades) '
      f'-> using {"log" if use_log_x else "linear"} x-axis')

rho, p_value = scipy.stats.spearmanr(conditioning_df['intercept_snr'], conditioning_df['abs_delta_tau'])
print(f'Spearman(intercept_SNR, |Delta_tau|) = {rho:.3f} (p={p_value:.3f})')


# ── Plot 1: scatter, |Delta_tau| vs intercept_SNR ─────────────────────────────

hemisphere_color = {'n': '#1f77b4', 's': '#d62728'}
variant_marker = {'native': 'o', 'va': '^'}

fig, ax = plt.subplots(figsize=(9, 7))
for hemisphere in HEMISPHERES:
    for variant in va_str_dict.keys():
        sub = conditioning_df[(conditioning_df['hemisphere'] == hemisphere) & (conditioning_df['variant'] == variant)]
        ax.scatter(sub['intercept_snr'], sub['abs_delta_tau'], color=hemisphere_color[hemisphere],
                   marker=variant_marker[variant], s=50, edgecolor='black', linewidth=0.4,
                   label=f'{hemisphere}H ({variant})')

known_point = conditioning_df[(conditioning_df['hemisphere'] == 's') & (conditioning_df['season'] == 'DJF') & (conditioning_df['variant'] == 'va')].iloc[0]
ax.annotate('SH DJF (va)\n(originally diagnosed outlier)',
            xy=(known_point['intercept_snr'], known_point['abs_delta_tau']),
            xytext=(20, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

if use_log_x:
    ax.set_xscale('log')
ax.set_xlabel('intercept SNR = |Re-intercept| / standard_error(Re-intercept)')
ax.set_ylabel('|Delta_tau| = |tau_autocorr_efold - tau_fit_3| (days)')
ax.set_title('Does tau_fit_3\'s intercept SNR predict its disagreement with tau_autocorr_efold?\n'
              '(tests the original SH DJF mechanism directly: small intercept relative to its own noise)')
ax.grid(True, which='both' if use_log_x else 'major')
ax.legend(fontsize=8)

ax.text(0.03, 0.97, f'Spearman rho = {rho:.3f} (p = {p_value:.3f})\nn = {len(conditioning_df)}',
        transform=ax.transAxes, fontsize=10, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
scatter_file = os.path.join(plot_dir, 'tau_fit_3_conditioning_snr_scatter.png')
plt.savefig(scatter_file)
plt.close(fig)
print(f'Saved figure to {scatter_file}')


# ── Plot 2: bar chart, styled to match tau_fit_3_conditioning_{hemisphere}hemisphere.png ──

variant_hatch = {'native': '', 'va': '//'}

for hemisphere in HEMISPHERES:
    sub = conditioning_df[conditioning_df['hemisphere'] == hemisphere]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(SEASONS))
    bar_width = 0.35

    for variant, offset in [('native', -0.5 * bar_width), ('va', 0.5 * bar_width)]:
        variant_rows = sub[sub['variant'] == variant].set_index('season').loc[SEASONS]
        ax.bar(x + offset, variant_rows['intercept_snr'].values, width=bar_width,
               color='#ff7f0e', hatch=variant_hatch[variant], edgecolor='black',
               linewidth=0.5, label=f'{variant}')

    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel('intercept SNR (|intercept| / stderr)')
    ax.set_xlabel('season')
    ax.set_title(f'{hemisphere}H: tau_fit_3 regression conditioning (intercept SNR) - '
                 'solid=native, hatched=va')
    ax.grid(True, axis='y')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(plot_dir, f'tau_fit_3_conditioning_snr_{hemisphere}hemisphere.png')
    plt.savefig(out_file)
    plt.close(fig)
    print(f'Saved figure to {out_file}')

print('Done.')
