"""
Compares the original cospectral tau (tau_fit_3, from jra55_850's power_spec.nc)
against the new season-boundary-respecting autocorrelation e-fold tau (from
jra_simpson_lag's tau_simpson_sliding.nc, eff.compute_simpson_sliding_tau) - see
/home/links/ct715/.claude/plans/wobbly-prancing-giraffe.md for the lag_method=
'simpson_sliding' background, and functions/SIT_functions/SIT_eddy_feedback_functions.py:1163
for how the new tau is computed.

Both tau_fit_3 and tau_autocorr_efold estimate the same underlying SIT relaxation
timescale (du/dt = F - u/tau, F = div1_QG); they differ in domain (cospectral
phase vs. time-domain autocorrelation decay) and, critically, in whether the
segment/lag construction respects season/year boundaries. tau_fit_3 is computed
over an STFT segment length (256 days) that straddles year boundaries for any
season other than 'all_time' - see the conversation this script came out of for
the diagnosis. tau_autocorr_efold never compares two days from different years.
So for 'all_time' the two should be closely comparable (no boundary issue either
way); for the 12 rolling 3-month seasons, divergence between them is the
signature of the boundary artifact rather than genuine estimator disagreement.

Also pulls the eddy-forcing (div1_QG) autocorrelation e-fold alongside the wind
(ucomp) one - it's not compared against tau_fit_3 (no equivalent exists there),
but it's the LH01-precondition diagnostic from the same conversation: it should
be short relative to the wind e-fold if the linear relaxation model's assumption
(forcing approximately white relative to the response timescale) holds.

Scope: both hemispheres (unlike method_comparison.py's southern-hemisphere-only),
all_time + all 12 rolling 3-month seasons (unlike method_comparison.py's
all_time/DJF/JJA), native and va variants. Only div1_QG (not div1_QG_123/gt3),
matching b-parameter/b_methodology/tau_values/extract_tau_fit_3_jra55.py.

Mirrors, in structure (script_dir-relative data/plots dirs, npz save-and-print),
method_comparison.py in the same all_plots_true/ directory - but saves one
combined table across all season/hemisphere/variant combos rather than one
figure per time_frame, since this is a scalar comparison, not spectral/lag curves.
"""

import os

import numpy as np
import pandas as pd
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

old_power_spec_file = os.path.join(
    all_plots_true_dir, 'jra55_850_sit_plots', '1979_2014', '6hourly',
    'level_250_500_850hPa', 'power_spec.nc')
new_tau_file = os.path.join(
    all_plots_true_dir, 'jra_simpson_lag_sit_plots', '1979_2014', '6hourly',
    'level_250_500_850hPa', 'tau_simpson_sliding.nc')

old_ds = xar.open_dataset(old_power_spec_file, auto_complex=True)
new_ds = xar.open_dataset(new_tau_file)


def old_tau_fit_3(va_str, hemisphere, season):
    ucomp = f'ucomp{va_str}'
    div1 = f'div1_QG{va_str}'
    name = (f'{div1}_PCs_from_{ucomp}_{hemisphere}_{season}_'
            f'{ucomp}_PCs_{hemisphere}_{season}_phase_diff_tau_fit_3')
    return float(old_ds[name].values)


def new_tau_autocorr_efold(var_stem, va_str, hemisphere, season):
    name = f'{var_stem}{va_str}_tau_autocorr_efold_{hemisphere}_{season}'
    return float(new_ds[name].values)


# ── Build the combined comparison table ───────────────────────────────────────

rows = []
for hemisphere in HEMISPHERES:
    for season in SEASONS:
        for va_label, va_str in va_str_dict.items():
            tau_old = old_tau_fit_3(va_str, hemisphere, season)
            tau_new_ucomp = new_tau_autocorr_efold('ucomp', va_str, hemisphere, season)
            tau_new_div1 = new_tau_autocorr_efold('div1_QG', va_str, hemisphere, season)

            rows.append({
                'dataset': 'jra55',
                'hemisphere': hemisphere,
                'season': season,
                'variant': va_label,
                'tau_fit_3_original': tau_old,
                'tau_autocorr_efold_ucomp_simpson': tau_new_ucomp,
                'tau_autocorr_efold_div1_QG_simpson': tau_new_div1,
                'diff_new_minus_old': tau_new_ucomp - tau_old,
                'pct_diff': 100. * (tau_new_ucomp - tau_old) / tau_old,
            })

comparison_df = pd.DataFrame(rows)

data_dir = os.path.join(script_dir, 'data')
plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

csv_file = os.path.join(data_dir, 'tau_comparison.csv')
comparison_df.to_csv(csv_file, index=False)
print(f'Saved data to {csv_file}')

npz_file = os.path.join(data_dir, 'tau_comparison.npz')
np.savez(
    npz_file,
    hemisphere=comparison_df['hemisphere'].values,
    season=comparison_df['season'].values,
    variant=comparison_df['variant'].values,
    tau_fit_3_original=comparison_df['tau_fit_3_original'].values,
    tau_autocorr_efold_ucomp_simpson=comparison_df['tau_autocorr_efold_ucomp_simpson'].values,
    tau_autocorr_efold_div1_QG_simpson=comparison_df['tau_autocorr_efold_div1_QG_simpson'].values,
)
print(f'Saved data to {npz_file}')

print(comparison_df.sort_values('pct_diff', key=np.abs, ascending=False).head(10).to_string(index=False))


# ── Comparison figure: one per hemisphere ─────────────────────────────────────
#
# Top panel: tau_fit_3 (original, cospectral) vs tau_autocorr_efold (new,
# season-boundary-respecting), grouped by season, paired native/va bars per
# source (color = source, matching method_comparison.py's color=method
# convention; hatch = variant, standing in for that script's linestyle=variant
# since these are bars, not lines).
# Bottom panel: div1_QG (eddy forcing) autocorrelation e-fold alone, on its own
# scale (~1 day vs ~5-15 days for the others - would be flattened on the same
# axis) - the LH01-precondition diagnostic, not a comparison against tau_fit_3.

source_color = {'tau_fit_3_original': '#1f77b4', 'tau_autocorr_efold_ucomp_simpson': '#d62728'}
source_label = {'tau_fit_3_original': 'Original (tau_fit_3, cospectral)',
                 'tau_autocorr_efold_ucomp_simpson': 'Simpson sliding (autocorr e-fold)'}
variant_hatch = {'native': '', 'va': '//'}

for hemisphere in HEMISPHERES:
    sub = comparison_df[comparison_df['hemisphere'] == hemisphere]

    fig, (ax_tau, ax_div1) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    n_seasons = len(SEASONS)
    x = np.arange(n_seasons)
    bar_width = 0.2

    # 4 bars per season: {original, simpson} x {native, va}
    slots = [
        ('tau_fit_3_original', 'native', x - 1.5 * bar_width),
        ('tau_fit_3_original', 'va', x - 0.5 * bar_width),
        ('tau_autocorr_efold_ucomp_simpson', 'native', x + 0.5 * bar_width),
        ('tau_autocorr_efold_ucomp_simpson', 'va', x + 1.5 * bar_width),
    ]

    seen_labels = set()
    for source_col, variant, offsets in slots:
        variant_rows = sub[sub['variant'] == variant].set_index('season').loc[SEASONS]
        label = source_label[source_col] if source_col not in seen_labels else None
        seen_labels.add(source_col)
        ax_tau.bar(offsets, variant_rows[source_col].values, width=bar_width,
                   color=source_color[source_col], hatch=variant_hatch[variant],
                   edgecolor='black', linewidth=0.5, label=label)

    ax_tau.set_xticks(x)
    ax_tau.set_xticklabels(SEASONS)
    ax_tau.set_ylabel('tau (days)')
    ax_tau.set_title(f'{hemisphere}H: original (tau_fit_3) vs Simpson sliding (autocorr e-fold) '
                      '- solid=native, hatched=va')
    ax_tau.grid(True, axis='y')
    ax_tau.legend(fontsize=8)

    for variant, offset in [('native', -0.5 * bar_width), ('va', 0.5 * bar_width)]:
        ax_div1.bar(x + offset, sub[sub['variant'] == variant].set_index('season').loc[SEASONS, 'tau_autocorr_efold_div1_QG_simpson'].values,
                   width=bar_width, color='#2ca02c', hatch=variant_hatch[variant],
                   edgecolor='black', linewidth=0.5, label=f'div1_QG e-fold ({variant})')

    ax_div1.set_xticks(x)
    ax_div1.set_xticklabels(SEASONS)
    ax_div1.set_ylabel('tau (days)')
    ax_div1.set_xlabel('season')
    ax_div1.set_title('Eddy forcing (div1_QG) autocorrelation e-fold - LH01 "forcing approximately '
                       'white" precondition check, not comparable to the panel above')
    ax_div1.grid(True, axis='y')
    ax_div1.legend(fontsize=8)

    plt.tight_layout()

    out_file = os.path.join(plot_dir, f'tau_comparison_{hemisphere}hemisphere.png')
    plt.savefig(out_file)
    plt.close(fig)
    print(f'Saved figure to {out_file}')

print('Done.')
