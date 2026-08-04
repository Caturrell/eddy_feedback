"""
Variant of z_fig5_250-500-850_va-vs-native.py: keeps only panel [1,1] of that
figure (the b parameter, lag 7-14) and repeats it side-by-side for both level
configurations, instead of just the 3-level (250,500,850hPa) subset.

Source data:
    b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/
        6hourly/level_250_500_850hPa/b_dataset.nc                (b parameter, lag 7-14)
        6hourly/level_full_100_850/b_dataset.nc                  (b parameter, lag 7-14)

Southern hemisphere, all_time.
    (a) left:  b parameter, level_250_500_850hPa (matches panel [1,1] /
               ax_b of z_fig5_250-500-850_va-vs-native.py)
    (b) right: b parameter, level_full_100_850

Solid lines: va variant, matching
    level_*/b_plots/s_hemisphere/all_time/_va/ucomp_va_div1_QG_gt3_va_b_s.pdf
Dashed lines: native (non-va) variant, matching
    level_*/b_plots/s_hemisphere/all_time/ucomp_div1_QG_gt3_b_s.pdf
"""

import os
import numpy as np
import xarray as xar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.abspath(__file__))

level_dirs = {
    '250_500_850hPa': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'level_250_500_850hPa'
    ),
    'full_100_850': os.path.join(
        script_dir, 'all_plots_true', 'jra55_850_sit_plots', '1979_2014',
        '6hourly', 'level_full_100_850'
    ),
}
level_labels = {'250_500_850hPa': '250,500,850hPa', 'full_100_850': '100-850hPa'}

hemisphere = 's'
time_frame = 'all_time'

vars_to_analyse = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']
var_colors = {'div1_QG': 'tab:blue', 'div1_QG_123': 'tab:orange', 'div1_QG_gt3': 'tab:green'}
var_labels = {'div1_QG': 'all k', 'div1_QG_123': 'k1-3', 'div1_QG_gt3': 'k>3'}

variant_va_str = {'va': '_va', 'native': ''}
variant_linestyles = {'va': '-', 'native': '--'}
variant_labels = {'va': 'vertically-averaged', 'native': 'non-averaged'}


def extract_b(b_ds, var_to_analyse, va_str):
    name = f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'
    return b_ds['lag'].values, b_ds[name].values


b_ds_dict = {level_key: xar.open_dataset(os.path.join(d, 'b_dataset.nc')) for level_key, d in level_dirs.items()}

# ── Compute ──────────────────────────────────────────────────────────────────

b_results = {}
for level_key, b_ds in b_ds_dict.items():
    for var_to_analyse in vars_to_analyse:
        for variant_key, va_str in variant_va_str.items():
            b_lag, b_val = extract_b(b_ds, var_to_analyse, va_str)
            b_results[(level_key, var_to_analyse, variant_key)] = (b_lag, b_val)

# ── Save data ────────────────────────────────────────────────────────────────

data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

save_dict = {}
for (level_key, var_to_analyse, variant_key), (b_lag, b_val) in b_results.items():
    save_dict[f'b_lag_{level_key}_{var_to_analyse}_{variant_key}'] = b_lag
    save_dict[f'b_val_{level_key}_{var_to_analyse}_{variant_key}'] = b_val

np.savez(os.path.join(data_dir, 'fig5a_compare_all_va_native_jra55.npz'), **save_dict)
print(f"Saved data to {data_dir}/fig5a_compare_all_va_native_jra55.npz")

# ── Plot ─────────────────────────────────────────────────────────────────────

plot_dir = os.path.join(script_dir, 'plots', 'z_extra_analysis')
os.makedirs(plot_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, level_key in zip(axes, level_dirs):
    for var_to_analyse in vars_to_analyse:
        color = var_colors[var_to_analyse]
        for variant_key in variant_va_str:
            linestyle = variant_linestyles[variant_key]
            b_lag, b_val = b_results[(level_key, var_to_analyse, variant_key)]
            ax.plot(b_lag, b_val, color=color, linestyle=linestyle)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlim(7, 14)
    ax.set_ylim(-0.15, 0.15)
    ax.grid(True)
    ax.set_xlabel('lag (days)')
    ax.set_ylabel(r'$b$')
    ax.set_title(f'Feedback strength, b (SH; {level_labels[level_key]})')

panel_labels = ['(a)', '(b)']
for ax, label in zip(axes, panel_labels):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left')

color_handles = [Line2D([0], [0], color=var_colors[v], linestyle='-', label=var_labels[v]) for v in vars_to_analyse]
linestyle_handles = [Line2D([0], [0], color='k', linestyle=variant_linestyles[k], label=variant_labels[k])
                     for k in variant_va_str]

color_legend = fig.legend(handles=color_handles, loc='upper center', bbox_to_anchor=(0.5, 0.06), ncol=3, fontsize=9, frameon=False)
fig.add_artist(color_legend)
fig.legend(handles=linestyle_handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.1, 1, 1])

out_file = os.path.join(plot_dir, 'z_fig5a_compare_all_va_native.png')
plt.savefig(out_file, bbox_inches='tight')
plt.close(fig)
print(f'Saved figure to {out_file}')
