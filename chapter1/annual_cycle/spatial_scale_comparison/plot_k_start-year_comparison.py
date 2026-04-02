import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypalettes import load_cmap

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/data/efp_cut_ends'

def load_mean(folder, ts):
    path = os.path.join(BASE, folder, ts, 'efp_results.json')
    with open(path) as f:
        return json.load(f)

def load_500(folder, ts):
    path = os.path.join(BASE, folder, ts, 'efp_results_500hPa.json')
    with open(path) as f:
        return json.load(f)

data = {
    '1958_2016': {
        'mean': {'6h':    load_mean('6hourly_efp', '1958_2016'),
                 'daily': load_mean('daily_efp',   '1958_2016')},
        '500':  {'6h':    load_500('500hPa_6hourly_efp', '1958_2016'),
                 'daily': load_500('500hPa_daily_efp',   '1958_2016')},
    },
    '1979_2016': {
        'mean': {'6h':    load_mean('6hourly_efp', '1979_2016'),
                 'daily': load_mean('daily_efp',   '1979_2016')},
        '500':  {'6h':    load_500('500hPa_6hourly_efp', '1979_2016'),
                 'daily': load_500('500hPa_daily_efp',   '1979_2016')},
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────
MONTH_MAP   = {1:'J',2:'F',3:'M',4:'A',5:'M',6:'J',
               7:'J',8:'A',9:'S',10:'O',11:'N',12:'D'}
MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
M2IDX       = {m: i for i, m in enumerate(MONTH_ORDER)}

def extract_series(d, key):
    pairs = [(M2IDX[info['months'][1]], info['efp'])
             for info in d[key].values()]
    pairs.sort()
    return zip(*pairs)

# ── style ─────────────────────────────────────────────────────────────────────
cmap   = load_cmap('highcontrast')
colors = {
    'all': cmap.colors[0],
    '123': cmap.colors[1],
    'gt3': cmap.colors[2],
}
styles = {'6h': '-', 'daily': '--'}

# ── Figure 1: 2x2 ─────────────────────────────────────────────────────────────
# rows: NH (top), SH (bottom) | cols: 1958-2016 (left), 1979-2016 (right)
# each panel: 6 lines (3 EFP types × 2 frequencies)

row_cfg = [
    ('Northern Hemisphere', 'efp_nh', 'efp_nh_123', 'efp_nh_gt3'),
    ('Southern Hemisphere', 'efp_sh', 'efp_sh_123', 'efp_sh_gt3'),
]
col_cfg = [
    ('1958–2016', '1958_2016', 'a', 'c'),
    ('1979–2016', '1979_2016', 'b', 'd'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row', sharex=True)

for row, (hemi, all_key, k123_key, gt3_key) in enumerate(row_cfg):
    for col, (ts_label, ts_key, label_t, label_b) in enumerate(col_cfg):
        ax = axes[row, col]
        panel_label = label_t if row == 0 else label_b
        ts_data = data[ts_key]

        for freq in ('6h', 'daily'):
            ls = styles[freq]

            xs, ys = extract_series(ts_data['mean'][freq], all_key)
            ax.plot(xs, ys, marker='o', linewidth=2, color=colors['all'], linestyle=ls)

            xs, ys = extract_series(ts_data['mean'][freq], k123_key)
            ax.plot(xs, ys, marker='o', linewidth=2, color=colors['123'], linestyle=ls)

            xs, ys = extract_series(ts_data['mean'][freq], gt3_key)
            ax.plot(xs, ys, marker='o', linewidth=2, color=colors['gt3'], linestyle=ls)

        ax.set_xticks(range(12))
        ax.set_xticklabels([MONTH_MAP[m] for m in MONTH_ORDER])
        ax.grid(True, axis='y', alpha=0.4)
        ax.tick_params(labelsize=10)
        ax.set_title(f'$\\bf{{({panel_label})}}$ {hemi} — {ts_label}', fontsize=13)

        if col == 0:
            ax.set_ylabel('EFP', fontsize=13)
        if row == 1:
            ax.set_xlabel('Month', fontsize=13)

legend_handles = [
    Line2D([0], [0], color=colors['all'], lw=2, label='All $k$'),
    Line2D([0], [0], color=colors['123'], lw=2, label='$k = 1$–$3$'),
    Line2D([0], [0], color=colors['gt3'], lw=2, label='$k > 3$'),
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='6-hourly'),
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='Daily'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=12)

plt.suptitle('EFP annual cycle by spatial scale: 1958–2016 vs 1979–2016', fontsize=14, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])

PLOTS = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/plots'
os.makedirs(PLOTS, exist_ok=True)
save_path = os.path.join(PLOTS, 'start_year_comparison_w-cut-ends.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")


# ── Figure 2: 6-panel ─────────────────────────────────────────────────────────
# rows: 6h (top), daily (bottom) | cols: EFP_pr, EFP_QG, EFP_QG_500
# colours: NH vs SH | solid: 1958-2016 | dashed: 1979-2016

hemi_colors = {'nh': cmap.colors[0], 'sh': cmap.colors[1]}

panel_cfg = [
    (0, 0, 'efp_nh',     'efp_sh',     'All $k$',     'a'),
    (0, 1, 'efp_nh_123', 'efp_sh_123', '$k = 1$–$3$', 'b'),
    (0, 2, 'efp_nh_gt3', 'efp_sh_gt3', '$k > 3$',     'c'),
    (1, 0, 'efp_nh',     'efp_sh',     'All $k$',     'd'),
    (1, 1, 'efp_nh_123', 'efp_sh_123', '$k = 1$–$3$', 'e'),
    (1, 2, 'efp_nh_gt3', 'efp_sh_gt3', '$k > 3$',     'f'),
]
freq_rows   = {0: '6h', 1: 'daily'}
freq_labels = {'6h': '6-hourly', 'daily': 'Daily'}

fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8), sharey='row', sharex=True)

for row, col, nh_key, sh_key, col_title, panel_label in panel_cfg:
    ax   = axes2[row, col]
    freq = freq_rows[row]

    for ts_key, ls in (('1958_2016', '--'), ('1979_2016', '-')):
        d = data[ts_key]['mean'][freq]
        for hemi, key in (('nh', nh_key), ('sh', sh_key)):
            xs, ys = extract_series(d, key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=hemi_colors[hemi], linestyle=ls)

    ax.set_xticks(range(12))
    ax.set_xticklabels([MONTH_MAP[m] for m in MONTH_ORDER])
    ax.grid(True, axis='y', alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.set_title(
        f'$\\bf{{({panel_label})}}$ {freq_labels[freq]} — {col_title}',
        fontsize=13
    )
    if col == 0:
        ax.set_ylabel('EFP', fontsize=13)
    if row == 1:
        ax.set_xlabel('Month', fontsize=13)

legend2_handles = [
    Line2D([0], [0], color=hemi_colors['nh'], lw=2, label='NH'),
    Line2D([0], [0], color=hemi_colors['sh'], lw=2, label='SH'),
    Line2D([0], [0], color='k', lw=2, linestyle='--',  label='1958–2016'),
    Line2D([0], [0], color='k', lw=2, linestyle='-', label='1979–2016'),
]
fig2.legend(handles=legend2_handles, loc='lower center', ncol=4,
            bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=12)

plt.suptitle('Start-year comparison by spatial scale and frequency', fontsize=14, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])

save_path2 = os.path.join(PLOTS, 'start_year_comparison_6panel_w-cut-ends.png')
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path2}")
