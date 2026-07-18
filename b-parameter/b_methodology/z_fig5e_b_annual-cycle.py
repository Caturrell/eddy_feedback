"""
Variant of z_fig5d_b_all-seasons.py: plots the mean-b-by-season panel only,
reordered onto an annual-cycle x-axis (starting at JJA / centred on July,
wrapping to MJJ / centred on June), matching the x-axis convention of:

    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/
        level_250_500_850hPa/b_plots/s_hemisphere/annual_cycle/_va/
            ucomp_va_div1_QG_gt3_b_annual_cycle.pdf

Each of the 12 overlapping 3-month seasons is centred on a single month
(DJF -> Jan, JFM -> Feb, ..., NDJ -> Dec); those centre months are used as
the single-letter x-axis labels, in the order J A S O N D J F M A M J.
Markers for each wavenumber band are connected across the full annual
cycle (no separate "ALL" category, matching the reference plot).

Southern hemisphere, level_full_100_850, "va" variant, same b_dataset.nc
as z_fig5d_b_all-seasons.py.
"""

import os
import numpy as np
import xarray as xar
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

level_dir = os.path.join(
    script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
    '6hourly', 'level_full_100_850'
)

hemisphere = 's'
va_str = '_va'

# Seasons reordered onto an annual cycle starting at JJA (centred on July),
# with each season's centre-month letter as its x-axis label.
annual_cycle_seasons = ['JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ']
annual_cycle_labels = ['J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J']

vars_to_analyse = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
var_colors = {'div1_QG': 'tab:blue', 'div1_QG_123': 'tab:orange', 'div1_QG_gt3': 'tab:green'}
var_labels = {'div1_QG': 'div1_QG', 'div1_QG_123': 'div1_QG_123', 'div1_QG_gt3': 'div1_QG_gt3'}


def extract_b(b_ds, var_to_analyse, time_frame):
    name = f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'
    return b_ds['lag'].values, b_ds[name].values


b_ds = xar.open_dataset(os.path.join(level_dir, 'b_dataset.nc'))

# ── Compute ──────────────────────────────────────────────────────────────────

b_results = {}
for var_to_analyse in vars_to_analyse:
    for time_frame in annual_cycle_seasons:
        b_results[(var_to_analyse, time_frame)] = extract_b(b_ds, var_to_analyse, time_frame)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {}
for var_to_analyse in vars_to_analyse:
    for time_frame in annual_cycle_seasons:
        b_lag, b_val = b_results[(var_to_analyse, time_frame)]
        save_dict[f'b_lag_{var_to_analyse}_{time_frame}'] = b_lag
        save_dict[f'b_val_{var_to_analyse}_{time_frame}'] = b_val

np.savez(os.path.join(data_dir, 'b-parameter_annual-cycle_jra55.npz'), **save_dict)
print(f"Saved data to {data_dir}/b-parameter_annual-cycle_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

plot_dir = os.path.join(script_dir, 'plots', 'z_extra_analysis')
os.makedirs(plot_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))

for var_to_analyse in vars_to_analyse:
    color = var_colors[var_to_analyse]
    season_means = [np.nanmean(b_results[(var_to_analyse, tf)][1]) for tf in annual_cycle_seasons]
    ax.plot(range(len(annual_cycle_seasons)), season_means, marker='x', color=color,
            markersize=6, markeredgewidth=1.5, linewidth=1.2, label=var_labels[var_to_analyse])

ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlim(-0.5, len(annual_cycle_seasons) - 0.5)
ax.set_xticks(range(len(annual_cycle_seasons)))
ax.set_xticklabels(annual_cycle_labels)
ax.set_ylim(-0.2, 0.2)
ax.grid(True, axis='y', alpha=0.5)
ax.set_title('Feedback strength, b (SH; 100-850hPa, va), annual cycle')
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()

out_file = os.path.join(plot_dir, 'z_fig5e_b_annual-cycle.png')
plt.savefig(out_file, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure to {out_file}')
