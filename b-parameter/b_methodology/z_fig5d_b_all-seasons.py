"""
Variant of fig5_m-regression_b-parameter.py: plots only panel (d) (the b
parameter), but for every season plus all_time, instead of comparing level
configurations or variants. Uses the "va" variant only, for all seasons.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/level_full_100_850/b_dataset.nc

Southern hemisphere, level_full_100_850, "va" variant. Each wavenumber band
(all k, k1-3, k>3) is colour-coded as in fig5_m-regression_b-parameter.py;
each of the 12 individual seasons (DJF, JFM, ..., NDJ) is plotted as a thin
line, and all_time is overplotted as a thicker line of the same colour, for
each wavenumber band.

Matching (all_time trace only; season traces are the same b_dataset.nc
variables for each season key instead of "all_time"):
    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/level_full_100_850/
        b_plots/s_hemisphere/all_time/_va/
            ucomp_va_div1_QG_gt3_va_b_s.pdf
"""

import os
import numpy as np
import xarray as xar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

level_dir = os.path.join(
    script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
    '6hourly', 'level_full_100_850'
)

hemisphere = 's'
va_str = '_va'

season_list = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
all_time_season_list = season_list + ['all_time']

vars_to_analyse = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
var_colors = {'div1_QG': 'tab:blue', 'div1_QG_123': 'tab:orange', 'div1_QG_gt3': 'tab:green'}
var_labels = {'div1_QG': 'all k', 'div1_QG_123': 'k1-3', 'div1_QG_gt3': 'k>3'}


def extract_b(b_ds, var_to_analyse, time_frame):
    name = f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'
    return b_ds['lag'].values, b_ds[name].values


b_ds = xar.open_dataset(os.path.join(level_dir, 'b_dataset.nc'))

# ── Compute ──────────────────────────────────────────────────────────────────

b_results = {}
for var_to_analyse in vars_to_analyse:
    for time_frame in all_time_season_list:
        b_results[(var_to_analyse, time_frame)] = extract_b(b_ds, var_to_analyse, time_frame)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {}
for var_to_analyse in vars_to_analyse:
    for time_frame in all_time_season_list:
        b_lag, b_val = b_results[(var_to_analyse, time_frame)]
        save_dict[f'b_lag_{var_to_analyse}_{time_frame}'] = b_lag
        save_dict[f'b_val_{var_to_analyse}_{time_frame}'] = b_val

np.savez(os.path.join(data_dir, 'b-parameter_all-seasons_jra55.npz'), **save_dict)
print(f"Saved data to {data_dir}/b-parameter_all-seasons_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

plot_dir = os.path.join(script_dir, 'plots', 'z_extra_analysis')
os.makedirs(plot_dir, exist_ok=True)

fig, (ax_b, ax_cat) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for var_to_analyse in vars_to_analyse:
    color = var_colors[var_to_analyse]
    for time_frame in season_list:
        b_lag, b_val = b_results[(var_to_analyse, time_frame)]
        ax_b.plot(b_lag, b_val, color=color, linewidth=0.8, alpha=0.6)

    b_lag, b_val = b_results[(var_to_analyse, 'all_time')]
    ax_b.plot(b_lag, b_val, color=color, linewidth=3)

ax_b.axhline(0, color='k', linewidth=0.5)
ax_b.set_xlim(7, 14)
ax_b.set_ylim(-0.1, 0.1)
ax_b.grid(True)
ax_b.set_xlabel('lag (days)')
ax_b.set_ylabel(r'$b$')
ax_b.set_title('Feedback strength, b (SH; 100-850hPa, va), all seasons')

# ── Right panel: mean b (lag 7-14) per season, as coloured cross markers ─────

category_labels = season_list + ['ALL']
for var_to_analyse in vars_to_analyse:
    color = var_colors[var_to_analyse]

    season_means = [np.nanmean(b_results[(var_to_analyse, tf)][1]) for tf in season_list]
    ax_cat.plot(range(len(season_list)), season_means, marker='x', color=color,
                markersize=6, markeredgewidth=1.5, linewidth=1.2)

    all_time_mean = np.nanmean(b_results[(var_to_analyse, 'all_time')][1])
    ax_cat.plot(len(season_list), all_time_mean, marker='x', color=color,
                markersize=10, markeredgewidth=3)

ax_cat.axhline(0, color='k', linewidth=0.5)
ax_cat.axvline(len(season_list) - 0.5, color='grey', linestyle=':', linewidth=0.8)
ax_cat.set_xlim(-0.5, len(all_time_season_list) - 0.5)
ax_cat.set_xticks(range(len(all_time_season_list)))
ax_cat.set_xticklabels(category_labels, rotation=45, fontsize=8)
ax_cat.grid(axis='y', alpha=0.5)
ax_cat.set_xlabel('season')
ax_cat.set_title('Mean b (lag 7-14) by season')

color_handles = [Line2D([0], [0], color=var_colors[v], linestyle='-', label=var_labels[v]) for v in vars_to_analyse]
weight_handles = [Line2D([0], [0], color='k', linewidth=0.8, alpha=0.6, label='individual seasons'),
                  Line2D([0], [0], color='k', linewidth=3, label='all_time')]

color_legend = fig.legend(handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9, frameon=False)
fig.add_artist(color_legend)
fig.legend(handles=weight_handles, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])

out_file = os.path.join(plot_dir, 'z_fig5d_b_all-seasons.png')
plt.savefig(out_file, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure to {out_file}')
