"""
Compares the b-parameter annual cycle between the two lag methodologies:

    jra55_850  - original lag_method (all_plots_true/jra55_850_sit_plots)
    sliding_b  - Simpson-style sliding-segment lag_method='simpson_sliding'
                 (all_plots_true/sliding_b_sit_plots), same underlying
                 1979-2014 non-detrended anomalies, only the lag construction
                 differs (see jra55_250-850_calc_efp_b.py)

Source data: the 4 CSVs written by
    data_collection/collect_b_annual_cycle_comparison.py
to data_collection/data/sliding_vs_fixed_window/, one per level config
(level_250_500_850hPa, level_full_100_850) x b variant (va =
vertically-averaged, native = non-averaged), each holding both methods
distinguished by a 'method' column.

One figure per (level config, variant) - 4 total. Each figure plots the
12-season annual cycle (JJA-start, wrapping to MJJ, matching the x-axis
convention of z_fig5e_b_annual-cycle.py) for all three wavenumber bands
(div1_QG, div1_QG_123, div1_QG_gt3), colour-coded by band and line-styled by
method (solid = jra55_850, dashed = sliding_b). A separate grouped-bar
column past a dashed divider shows the all_time b value for each
band/method combination. All four figures share the same y-axis limits,
set from whichever (level config, variant) combination has the widest data
range, so the panels are directly comparable.

Southern hemisphere only.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data_collection', 'data', 'sliding_vs_fixed_window')

level_tags = ['250_500_850hPa', 'full_100_850']
level_titles = {'250_500_850hPa': '250/500/850hPa', 'full_100_850': 'full 100-850hPa'}

variant_tags = ['va', 'native']
variant_titles = {'va': 'vertically-averaged', 'native': 'non-averaged'}

methods = ['jra55_850', 'sliding_b']
method_linestyles = {'jra55_850': '-', 'sliding_b': '--'}
method_markers = {'jra55_850': 'x', 'sliding_b': 'o'}
method_hatches = {'jra55_850': '', 'sliding_b': '///'}
method_labels = {'jra55_850': 'jra55_850 (original)', 'sliding_b': 'sliding_b (Simpson sliding-lag)'}

vars_to_analyse = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
var_colors = {'div1_QG': 'tab:blue', 'div1_QG_123': 'tab:orange', 'div1_QG_gt3': 'tab:green'}
var_labels = {'div1_QG': 'all $k$', 'div1_QG_123': '$k$=1-3', 'div1_QG_gt3': '$k$>3'}

annual_cycle_seasons = ['JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ',
                         'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ']
centre_month_labels = ['J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J']

n_seasons = len(annual_cycle_seasons)
divider_x = n_seasons - 0.5 + 0.75
all_time_x = n_seasons + 1.25

plot_dir = os.path.join(script_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# ── Pass 1: load all combos and find the shared y-range ────────────────────

combo_data = {}
global_min, global_max = None, None

for level_tag in level_tags:
    for variant_tag in variant_tags:
        csv_path = os.path.join(data_dir, f'b_{level_tag}_{variant_tag}_s_annual_cycle.csv')
        combined_df = pd.read_csv(csv_path)

        dfs = {
            method: combined_df[combined_df['method'] == method].set_index('time_frame')
            for method in methods
        }
        combo_data[(level_tag, variant_tag)] = dfs

        for method in methods:
            df = dfs[method]
            combo_min = df[vars_to_analyse].values.min()
            combo_max = df[vars_to_analyse].values.max()
            global_min = combo_min if global_min is None else min(global_min, combo_min)
            global_max = combo_max if global_max is None else max(global_max, combo_max)

y_pad = 0.1 * (global_max - global_min)
shared_ylim = (global_min - y_pad, global_max + y_pad)

# ── Pass 2: plot each combo with the shared y-axis limits ──────────────────

for level_tag in level_tags:
    for variant_tag in variant_tags:

        dfs = combo_data[(level_tag, variant_tag)]

        fig, ax = plt.subplots(figsize=(9, 5.5))

        for var_to_analyse in vars_to_analyse:
            color = var_colors[var_to_analyse]
            for method in methods:
                df = dfs[method]
                season_vals = df.loc[annual_cycle_seasons, var_to_analyse].values
                ax.plot(range(n_seasons), season_vals,
                        color=color, linestyle=method_linestyles[method],
                        marker=method_markers[method], markersize=6,
                        markeredgewidth=1.5, linewidth=1.2)

        # ── All-time grouped-bar column ─────────────────────────────────────
        n_vars = len(vars_to_analyse)
        n_methods = len(methods)
        group_width = 1.6
        bar_width = group_width / (n_vars * n_methods)
        group_left = all_time_x - group_width / 2

        bar_idx = 0
        for var_to_analyse in vars_to_analyse:
            color = var_colors[var_to_analyse]
            for method in methods:
                df = dfs[method]
                all_time_val = df.loc['all_time', var_to_analyse]
                bar_x = group_left + (bar_idx + 0.5) * bar_width
                ax.bar(bar_x, all_time_val, width=bar_width * 0.9, color=color,
                       hatch=method_hatches[method], edgecolor='k', linewidth=0.6)
                bar_idx += 1

        ax.axvline(divider_x, color='0.5', linestyle=':', linewidth=1)

        ax.set_ylim(shared_ylim)

        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlim(-0.5, all_time_x + group_width / 2 + 0.3)
        ax.set_xticks(list(range(n_seasons)) + [all_time_x])
        ax.set_xticklabels(centre_month_labels + ['All-time'])
        ax.set_ylabel(r'$b$')
        ax.grid(True, axis='y', alpha=0.5)
        ax.set_title(
            f'Feedback strength, b (SH; {level_titles[level_tag]}, '
            f'{variant_titles[variant_tag]}): jra55_850 vs sliding_b'
        )

        color_handles = [Line2D([0], [0], color=var_colors[v], linewidth=2, label=var_labels[v])
                          for v in vars_to_analyse]
        method_handles = [Line2D([0], [0], color='k', linestyle=method_linestyles[m],
                                  marker=method_markers[m], markersize=6, markeredgewidth=1.5,
                                  label=method_labels[m])
                           for m in methods]

        color_legend = ax.legend(handles=color_handles, loc='upper left', fontsize=9,
                                  title='wavenumber band', title_fontsize=9)
        ax.add_artist(color_legend)
        ax.legend(handles=method_handles, loc='upper right', fontsize=9, title='method', title_fontsize=9)

        plt.tight_layout()

        out_file = os.path.join(plot_dir, f'{level_tag}_{variant_tag}_annual_cycle_comparison.png')
        plt.savefig(out_file, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'Saved figure to {out_file}')
