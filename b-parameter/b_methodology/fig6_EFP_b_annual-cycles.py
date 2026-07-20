"""
Variant of z_fig5d_b_all-seasons.py: compares the annual cycles of the b
parameter and the (500hPa) eddy feedback parameter (EFP), reordered onto an
annual-cycle x-axis (starting at JJA / centred on July, wrapping to MJJ /
centred on June), matching the x-axis convention of:

    all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/
        level_250_500_850hPa/b_plots/s_hemisphere/annual_cycle/_va/
            ucomp_va_div1_QG_gt3_b_annual_cycle.pdf

Each of the 12 overlapping 3-month seasons is centred on a single month
(DJF -> Jan, JFM -> Feb, ..., NDJ -> Dec); those centre months are used as
the single-letter x-axis labels, in the order J A S O N D J F M A M J.
Markers for each wavenumber band are connected across the full annual
cycle (no separate "ALL" category, matching the reference plot).

b: Southern hemisphere, level_250_500_850hPa (pressure-weighted, 3-level
250/500/850hPa config), same b_dataset.nc as z_fig5d_b_all-seasons.py (but
level_250_500_850hPa rather than level_full_100_850). Compared across all
three b variants stored in that file:
    va     - pressure-weighted vertical average of the 3 levels before EOF
    native - joint EOF across the 3-level x lat field (no _va/_500 suffix)
    _500   - single 500hPa level only

EFP: Southern hemisphere, 500hPa, from
    cmip6_b/efp_vs_b/jra55_efp_annual_cycle.csv

Plots for all three b variants are saved under plots/fig6_extras/<variant>/
for inspection; once a final variant is chosen, its plots should move to
plots/ directly.
"""

import json
import os
import numpy as np
import pandas as pd
import xarray as xar
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

level_dir = os.path.join(
    script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
    '6hourly', 'level_250_500_850hPa'
)

efp_csv_path = os.path.join(
    script_dir, '..', 'cmip6_b', 'efp_vs_b', 'jra55_efp_annual_cycle.csv'
)

efp_annual_json_path = os.path.join(
    script_dir, '..', 'EFP_extra_calcs', 'data', '1979_2014', 'efp_results_500hPa.json'
)

hemisphere = 's'

# Seasons reordered onto an annual cycle starting at JJA (centred on July),
# with each season's centre-month letter as its x-axis label.
annual_cycle_seasons = ['JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ']
annual_cycle_labels = ['J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J']

vars_to_analyse = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
var_colors = {'div1_QG': 'tab:blue', 'div1_QG_123': 'tab:orange', 'div1_QG_gt3': 'tab:green'}
var_labels = {'div1_QG': 'all $k$', 'div1_QG_123': '$k=1$-3', 'div1_QG_gt3': '$k>3$'}

# (va_str used in b_dataset.nc variable names, output subfolder name, plot label)
b_variant_configs = [
    ('_va', 'va', 'va (250/500/850hPa pressure-weighted mean)'),
    ('', 'native', 'native (joint EOF, 250/500/850hPa x lat)'),
    ('_500', '_500', '500hPa only'),
]

# Each season gets its own colour + marker shape (used by the EFP-vs-b scatter plots).
# Calendar order (DJF first) with a rainbow colour progression through the year.
season_calendar_order = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
season_marker_list = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', '<', '>', 'p']
season_color_list = plt.cm.rainbow(np.linspace(0, 1, len(season_calendar_order)))
season_color_map = dict(zip(season_calendar_order, season_color_list))
season_marker_map = dict(zip(season_calendar_order, season_marker_list))


def extract_b(b_ds, var_to_analyse, time_frame, va_str):
    name = f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'
    return b_ds['lag'].values, b_ds[name].values


b_ds = xar.open_dataset(os.path.join(level_dir, 'b_dataset.nc'))

efp_df = pd.read_csv(efp_csv_path)
efp_df = efp_df[efp_df['hemisphere'] == hemisphere]
efp_lookup = {(row.variant, row.season): row.efp for row in efp_df.itertuples()}

efp_season_values = {
    var_to_analyse: [efp_lookup[(var_to_analyse, tf)] for tf in annual_cycle_seasons]
    for var_to_analyse in vars_to_analyse
}

# Annual-mean EFP (SH, 500hPa) - plotted as a black cross on the scatter plots,
# excluded from the r/p and regression-line calculations.
with open(efp_annual_json_path) as f:
    efp_annual_data = json.load(f)

efp_annual_key_map = {'div1_QG': 'efp_sh', 'div1_QG_123': 'efp_sh_123', 'div1_QG_gt3': 'efp_sh_gt3'}
efp_annual_mean = {v: efp_annual_data[efp_annual_key_map[v]]['ANN']['efp'] for v in vars_to_analyse}

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

extras_dir = os.path.join(script_dir, 'plots', 'fig6_extras')

x = range(len(annual_cycle_seasons))

for va_str, subfolder, variant_label in b_variant_configs:

    # ── Compute ──────────────────────────────────────────────────────────────

    b_results = {}
    for var_to_analyse in vars_to_analyse:
        for time_frame in annual_cycle_seasons:
            b_results[(var_to_analyse, time_frame)] = extract_b(b_ds, var_to_analyse, time_frame, va_str)

    b_annual_mean = {
        var_to_analyse: np.nanmean(extract_b(b_ds, var_to_analyse, 'all_time', va_str)[1])
        for var_to_analyse in vars_to_analyse
    }

    b_season_means = {
        var_to_analyse: [np.nanmean(b_results[(var_to_analyse, tf)][1]) for tf in annual_cycle_seasons]
        for var_to_analyse in vars_to_analyse
    }

    # ── Save data ────────────────────────────────────────────────────────────

    save_dict = {}
    for var_to_analyse in vars_to_analyse:
        for time_frame in annual_cycle_seasons:
            b_lag, b_val = b_results[(var_to_analyse, time_frame)]
            save_dict[f'b_lag_{var_to_analyse}_{time_frame}'] = b_lag
            save_dict[f'b_val_{var_to_analyse}_{time_frame}'] = b_val
        save_dict[f'efp_val_{var_to_analyse}'] = np.array(efp_season_values[var_to_analyse])

    npz_file = os.path.join(data_dir, f'b-parameter_annual-cycle_jra55_{subfolder}.npz')
    np.savez(npz_file, **save_dict)
    print(f'Saved data to {npz_file}')

    # ── Plot ─────────────────────────────────────────────────────────────────

    plot_dir = os.path.join(extras_dir, subfolder)
    os.makedirs(plot_dir, exist_ok=True)

    # --- Layout 1: single panel, twin y-axis (b solid/x, EFP dashed/o) ------

    fig, ax_b = plt.subplots(figsize=(7, 5))
    ax_efp = ax_b.twinx()

    for var_to_analyse in vars_to_analyse:
        color = var_colors[var_to_analyse]
        ax_b.plot(x, b_season_means[var_to_analyse], marker='x', color=color,
                  markersize=6, markeredgewidth=1.5, linewidth=1.2, linestyle='-')
        ax_efp.plot(x, efp_season_values[var_to_analyse], marker='o', color=color,
                    markersize=5, linewidth=1.2, linestyle='--', alpha=0.8)

    ax_b.axhline(0, color='k', linewidth=0.5)
    ax_b.set_xlim(-0.5, len(annual_cycle_seasons) - 0.5)
    ax_b.set_xticks(range(len(annual_cycle_seasons)))
    ax_b.set_xticklabels(annual_cycle_labels)
    ax_b.set_ylim(-0.15, 0.15)
    ax_b.set_ylabel(r'$b$')
    ax_efp.set_ylim(0, 0.3)
    ax_efp.set_ylabel('EFP (500hPa)')
    ax_b.grid(True, axis='y', alpha=0.5)
    ax_b.set_title(f'b ({variant_label}) vs EFP (500hPa), SH annual cycle')

    color_handles = [Line2D([0], [0], color=var_colors[v], label=var_labels[v]) for v in vars_to_analyse]
    style_handles = [Line2D([0], [0], color='k', marker='x', linestyle='-', label='b'),
                     Line2D([0], [0], color='k', marker='o', linestyle='--', label='EFP')]
    color_legend = ax_b.legend(handles=color_handles, loc='upper left', fontsize=8)
    ax_b.add_artist(color_legend)
    ax_b.legend(handles=style_handles, loc='upper right', fontsize=8)

    plt.tight_layout()

    out_file = os.path.join(plot_dir, 'fig6_EFP_b_annual-cycle_twinaxis.png')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure to {out_file}')

    # --- Layout 2: two side-by-side panels, shared x-axis -------------------

    fig, (ax_b2, ax_efp2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for var_to_analyse in vars_to_analyse:
        color = var_colors[var_to_analyse]
        ax_b2.plot(x, b_season_means[var_to_analyse], marker='x', color=color,
                   markersize=6, markeredgewidth=1.5, linewidth=1.2, label=var_labels[var_to_analyse])
        ax_efp2.plot(x, efp_season_values[var_to_analyse], marker='x', color=color,
                     markersize=6, markeredgewidth=1.5, linewidth=1.2, label=var_labels[var_to_analyse])

    ax_b2.axhline(0, color='k', linewidth=0.5)
    ax_b2.set_xlim(-0.5, len(annual_cycle_seasons) - 0.5)
    ax_b2.set_xticks(range(len(annual_cycle_seasons)))
    ax_b2.set_xticklabels(annual_cycle_labels)
    ax_b2.set_ylim(-0.15, 0.15)
    ax_b2.grid(True, axis='y', alpha=0.5)
    ax_b2.set_title(f'b (SH; {variant_label})')
    ax_b2.legend(loc='lower right', fontsize=9)

    ax_efp2.set_ylim(0, 0.3)
    ax_efp2.grid(True, axis='y', alpha=0.5)
    ax_efp2.set_title('EFP (SH; 500hPa)')

    plt.tight_layout()

    out_file = os.path.join(plot_dir, 'fig6_EFP_b_annual-cycle_twopanel.png')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure to {out_file}')

    # --- Layout 3: EFP vs b scatter, one subplot per spatial scale, seasons
    #     encoded by colour + marker shape -------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    for ax, var_to_analyse in zip(axes, vars_to_analyse):
        for idx, season in enumerate(annual_cycle_seasons):
            ax.scatter(b_season_means[var_to_analyse][idx], efp_season_values[var_to_analyse][idx],
                       color=season_color_map[season], marker=season_marker_map[season], s=80,
                       edgecolor='k', linewidth=0.5, zorder=3)

        ax.scatter(b_annual_mean[var_to_analyse], efp_annual_mean[var_to_analyse],
                   color='black', marker='x', s=100, linewidth=2, zorder=4)

        b_vals_with_ann = b_season_means[var_to_analyse] + [b_annual_mean[var_to_analyse]]
        efp_vals_with_ann = efp_season_values[var_to_analyse] + [efp_annual_mean[var_to_analyse]]

        r_value, p_value = scipy.stats.pearsonr(b_vals_with_ann, efp_vals_with_ann)
        ax.text(0.03, 0.97, f'r = {r_value:.2f}\np = {p_value:.3f}', transform=ax.transAxes,
               ha='left', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))

        sns.regplot(x=b_vals_with_ann, y=efp_vals_with_ann, ax=ax,
                   scatter=False, ci=None, truncate=False,
                   line_kws=dict(color='0.4', linestyle='--', linewidth=1.2, zorder=2))

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(0, 0.3)
        ax.set_xlabel(r'$b$')
        ax.set_title(var_labels[var_to_analyse])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('EFP (500hPa)')

    season_handles = [Line2D([0], [0], marker=season_marker_map[season], color=season_color_map[season], linestyle='None',
                             markersize=8, markeredgecolor='k', markeredgewidth=0.5, label=season)
                      for season in season_calendar_order]
    fig.legend(handles=season_handles, loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0, 0.94, 1])

    scatter_file = os.path.join(plot_dir, 'fig6_EFP_vs_b_scatter.png')
    plt.savefig(scatter_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure to {scatter_file}')

    # ── Main comparison figure (va variant only): annual cycle on top row, ─────
    # scatter plots on bottom row - both rows the same panel height, panels
    # lettered (a)-(e) in bold.
    if subfolder == 'va':

        def bold_title(letter, text):
            return rf'$\mathbf{{({letter})}}$ {text}'

        def build_main_comparison_figure(season_order, season_labels, out_file):
            b_local = {v: [np.nanmean(b_results[(v, tf)][1]) for tf in season_order] for v in vars_to_analyse}
            efp_local = {v: [efp_lookup[(v, tf)] for tf in season_order] for v in vars_to_analyse}

            fig_main = plt.figure(figsize=(16, 10))
            gs = fig_main.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.3, wspace=0.5)

            # --- Top row: annual cycle - b (cols 0:3), EFP (cols 3:6) --------

            ax_b_top = fig_main.add_subplot(gs[0, 0:3])
            ax_efp_top = fig_main.add_subplot(gs[0, 3:6])

            for var_to_analyse in vars_to_analyse:
                color = var_colors[var_to_analyse]
                ax_b_top.plot(range(len(season_order)), b_local[var_to_analyse], marker='x', color=color,
                              markersize=6, markeredgewidth=1.5, linewidth=1.2, label=var_labels[var_to_analyse])
                ax_efp_top.plot(range(len(season_order)), efp_local[var_to_analyse], marker='x', color=color,
                                markersize=6, markeredgewidth=1.5, linewidth=1.2)

            ax_b_top.axhline(0, color='k', linewidth=0.5)
            ax_b_top.set_xlim(-0.5, len(season_order) - 0.5)
            ax_b_top.set_xticks(range(len(season_order)))
            ax_b_top.set_xticklabels(season_labels)
            ax_b_top.set_ylim(-0.15, 0.15)
            ax_b_top.set_ylabel(r'$b$')
            ax_b_top.grid(True, axis='y', alpha=0.5)
            ax_b_top.set_title(bold_title('a', r'$b$-parameter annual cycle'))
            ax_b_top.legend(loc='lower right', fontsize=10)

            ax_efp_top.set_xlim(-0.5, len(season_order) - 0.5)
            ax_efp_top.set_xticks(range(len(season_order)))
            ax_efp_top.set_xticklabels(season_labels)
            ax_efp_top.set_ylim(0, 0.3)
            ax_efp_top.set_ylabel('EFP')
            ax_efp_top.grid(True, axis='y', alpha=0.5)
            ax_efp_top.set_title(bold_title('b', 'EFP annual cycle'))

            # --- Bottom row: EFP vs b scatter, one panel per spatial scale ----
            # (season ordering here is independent of the annual-cycle x-axis
            # above - always plotted DJF...NDJ, same 12 points either way)

            ax_scatter = [fig_main.add_subplot(gs[1, 0:2]), fig_main.add_subplot(gs[1, 2:4]), fig_main.add_subplot(gs[1, 4:6])]

            for letter, ax, var_to_analyse in zip(['c', 'd', 'e'], ax_scatter, vars_to_analyse):
                b_vals = [np.nanmean(b_results[(var_to_analyse, tf)][1]) for tf in season_calendar_order]
                efp_vals = [efp_lookup[(var_to_analyse, tf)] for tf in season_calendar_order]

                for season, b_val, efp_val in zip(season_calendar_order, b_vals, efp_vals):
                    ax.scatter(b_val, efp_val, color=season_color_map[season], marker=season_marker_map[season],
                              s=80, edgecolor='k', linewidth=0.5, zorder=3)

                ax.scatter(b_annual_mean[var_to_analyse], efp_annual_mean[var_to_analyse],
                          color='black', marker='x', s=100, linewidth=2, zorder=4)

                b_vals_with_ann = b_vals + [b_annual_mean[var_to_analyse]]
                efp_vals_with_ann = efp_vals + [efp_annual_mean[var_to_analyse]]

                r_value, p_value = scipy.stats.pearsonr(b_vals_with_ann, efp_vals_with_ann)
                ax.text(0.03, 0.97, f'r = {r_value:.2f}\np = {p_value:.3f}', transform=ax.transAxes,
                       ha='left', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))

                sns.regplot(x=b_vals_with_ann, y=efp_vals_with_ann, ax=ax,
                           scatter=False, ci=None, truncate=False,
                           line_kws=dict(color='0.4', linestyle='--', linewidth=1.2, zorder=2))

                ax.axhline(0, color='k', linewidth=0.5)
                ax.axvline(0, color='k', linewidth=0.5)
                ax.set_xlim(-0.15, 0.15)
                ax.set_ylim(0, 0.3)
                ax.set_xlabel(r'$b$')
                ax.set_title(bold_title(letter, var_labels[var_to_analyse]))
                ax.grid(True, alpha=0.3)

            ax_scatter[0].set_ylabel('EFP')

            season_handles = [Line2D([0], [0], marker=season_marker_map[season], color=season_color_map[season], linestyle='None',
                                     markersize=7, markeredgecolor='k', markeredgewidth=0.5, label=season)
                              for season in season_calendar_order]
            season_legend = ax_scatter[0].legend(handles=season_handles, loc='lower right', fontsize=9, frameon=True, framealpha=0.9)
            ax_scatter[0].add_artist(season_legend)

            annual_mean_handle = [Line2D([0], [0], marker='x', color='black', linestyle='None',
                                         markersize=8, markeredgewidth=2, label='all-time')]
            ax_scatter[0].legend(handles=annual_mean_handle, loc='lower left', fontsize=9, frameon=True, framealpha=0.9)

            plt.savefig(out_file, bbox_inches='tight')
            plt.close(fig_main)
            print(f'Saved combined figure to {out_file}')

        build_main_comparison_figure(season_calendar_order, season_calendar_order,
                                     os.path.join(script_dir, 'plots', 'fig6_EFP_b_comparison.png'))
