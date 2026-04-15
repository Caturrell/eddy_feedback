import matplotlib.pyplot as plt
import numpy as np
import os
import json

cmip6_data_path = '/home/links/ct715/eddy_feedback/chapter1/cmip6/historical_runs/data/1979_2014/6h'
reanalysis_path = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/data/500hPa_6hourly_efp/1979_2016/efp_results_500hPa.json'

models = sorted(os.listdir(cmip6_data_path))

CENTRAL_MONTH_DICT = {
    'DJF': 7, 'JFM': 8, 'FMA': 9, 'MAM': 10,
    'AMJ': 11, 'MJJ': 12, 'JJA': 1, 'JAS': 2,
    'ASO': 3, 'SON': 4, 'OND': 5, 'NDJ': 6,
}

all_seasons = sorted(CENTRAL_MONTH_DICT, key=CENTRAL_MONTH_DICT.get)
x = np.array([CENTRAL_MONTH_DICT[s] for s in all_seasons])

# Load all model data
data = {}
for model in models:
    json_path = os.path.join(cmip6_data_path, model, f'{model}_efp_1979_2014_CMIP6_hist_6h.json')
    with open(json_path) as f:
        data[model] = json.load(f)

with open(reanalysis_path) as f:
    reanalysis_data = json.load(f)

# Panel layout: top row NH, bottom row SH
panel_keys = [
    ('efp_nh',     'All $k$'),
    ('efp_nh_123', '$k \leq 3$'),
    ('efp_nh_gt3', '$k > 3$'),
    ('efp_sh',     'All $k$'),
    ('efp_sh_123', '$k \leq 3$'),
    ('efp_sh_gt3', '$k > 3$'),
]

cmap   = plt.colormaps['tab20'].resampled(len(models))
colors = [cmap(i) for i in range(len(models))]

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey='row')

legend_handles = []
legend_labels  = []

for idx, (key, title) in enumerate(panel_keys):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    for m_idx, model in enumerate(models):
        efp_values = [data[model][key][s]['efp'] for s in all_seasons]
        line, = ax.plot(x, efp_values, color=colors[m_idx], label=model, linewidth=1.5)
        if idx == 0:
            legend_handles.append(line)
            legend_labels.append(model)

    reanalysis_values = [reanalysis_data[key][s]['efp'] for s in all_seasons]
    line_r, = ax.plot(x, reanalysis_values, color='black', linewidth=2.5, label='JRA55', zorder=5)
    if idx == 0:
        legend_handles.append(line_r)
        legend_labels.append('JRA55')

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([s[1] for s in all_seasons], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax.yaxis.grid(True, linewidth=0.6, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    if col == 0:
        hemi = 'Northern Hemisphere' if row == 0 else 'Southern Hemisphere'
        ax.set_ylabel(f'$\\bf{{{hemi}}}$\nEFP', fontsize=10)

fig.legend(
    legend_handles, legend_labels,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    fontsize=7.5,
    frameon=True,
    ncol=1,
    title='Model',
    title_fontsize=9,
)

fig.suptitle('CMIP6 historical annual cycles compared to reanalysis', fontsize=14)
fig.tight_layout()

save_path = os.path.join(
    '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/all_seasons_underestimate/plots',
    'hist_cmip6_annual_cycles.png'
)
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
