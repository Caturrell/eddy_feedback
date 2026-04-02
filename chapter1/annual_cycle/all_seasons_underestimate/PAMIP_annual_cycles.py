import matplotlib.pyplot as plt
import numpy as np
import os
import json

pamip_data_path = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/data/PAMIP'
reanalysis_path = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/data/efp_cut_ends/daily_efp/1979_2016/efp_results.json'

models = sorted(os.listdir(pamip_data_path))

pamip_seasons = ['JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 'FMA', 'MAM']
all_seasons = pamip_seasons + ['AMJ', 'MJJ']
x = np.arange(len(all_seasons))

# Load all model data
data = {}
for model in models:
    json_path = os.path.join(pamip_data_path, model, 'efp_results.json')
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

colors = plt.cm.tab10(np.linspace(0, 0.8, len(models)))

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey='row')

for idx, (key, title) in enumerate(panel_keys):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    x_pamip = np.arange(len(pamip_seasons))
    for m_idx, model in enumerate(models):
        efp_values = [data[model][key][s]['efp'] for s in pamip_seasons]
        ax.plot(x_pamip, efp_values, color=colors[m_idx], label=model, linewidth=1.5)

    reanalysis_values = [reanalysis_data[key][s]['efp'] for s in all_seasons]
    ax.plot(x, reanalysis_values, color='black', linewidth=2.5, label='JRA55', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_seasons, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax.yaxis.grid(True, linewidth=0.6, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    if col == 0:
        hemi = 'NH' if row == 0 else 'SH'
        ax.set_ylabel(f'{hemi}  EFP', fontsize=10)

axes[0, 2].legend(loc='upper right', fontsize=7, framealpha=0.7)

fig.suptitle('PAMIP annual cycles compared to reanalysis', fontsize=14)
fig.tight_layout()

save_path = os.path.join(
    '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/all_seasons_underestimate/plots',
    'PAMIP_annual_cycles.png'
)
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
