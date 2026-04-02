import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypalettes import load_cmap

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/data'
BASE_CUT = os.path.join(BASE, 'efp_cut_ends')
TS       = '1979_2016'

def load_mean(base, folder, ts=TS):
    path = os.path.join(base, folder, ts, 'efp_results.json')
    with open(path) as f:
        return json.load(f)

def load_500(base, folder, ts=TS):
    path = os.path.join(base, folder, ts, 'efp_results_500hPa.json')
    with open(path) as f:
        return json.load(f)

data = {
    'no_cut': {
        'mean': {'6h':    load_mean(BASE,     '6hourly_efp'),
                 'daily': load_mean(BASE,     'daily_efp')},
        '500':  {'6h':    load_500(BASE,     '500hPa_6hourly_efp'),
                 'daily': load_500(BASE,     '500hPa_daily_efp')},
    },
    'cut': {
        'mean': {'6h':    load_mean(BASE_CUT, '6hourly_efp'),
                 'daily': load_mean(BASE_CUT, 'daily_efp')},
        '500':  {'6h':    load_500(BASE_CUT, '500hPa_6hourly_efp'),
                 'daily': load_500(BASE_CUT, '500hPa_daily_efp')},
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
    'all':  cmap.colors[0],
    '123':  cmap.colors[1],
    'gt3':  cmap.colors[2],
}
styles = {'6h': '-', 'daily': '--'}

# ── Figure 1: 2×2 — rows: NH/SH | cols: no_cut/cut ───────────────────────────
# colours: k-group | linestyles: frequency
row_cfg = [
    ('Northern Hemisphere', 'efp_nh', 'efp_nh_123', 'efp_nh_gt3'),
    ('Southern Hemisphere', 'efp_sh', 'efp_sh_123', 'efp_sh_gt3'),
]
col_cfg = [
    ('cut_ends=False', 'no_cut', 'a', 'b'),
    ('cut_ends=True',  'cut',    'c', 'd'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row', sharex=True)

for row, (hemi, all_key, k123_key, gt3_key) in enumerate(row_cfg):
    for col, (cut_title, cut_label, label_t, label_b) in enumerate(col_cfg):
        ax = axes[row, col]
        panel_label = label_t if row == 0 else label_b
        cut_data = data[cut_label]

        for freq in ('6h', 'daily'):
            ls = styles[freq]

            # all-k
            xs, ys = extract_series(cut_data['mean'][freq], all_key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=colors['all'], linestyle=ls)

            # k=1-3
            xs, ys = extract_series(cut_data['mean'][freq], k123_key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=colors['123'], linestyle=ls)

            # k>3
            xs, ys = extract_series(cut_data['mean'][freq], gt3_key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=colors['gt3'], linestyle=ls)

        ax.set_xticks(range(12))
        ax.set_xticklabels([MONTH_MAP[m] for m in MONTH_ORDER])
        ax.grid(True, axis='y', alpha=0.4)
        ax.tick_params(labelsize=10)
        ax.set_title(f'$\\bf{{({panel_label})}}$ {hemi} — {cut_title}', fontsize=13)

        if col == 0:
            ax.set_ylabel('EFP', fontsize=13)
        if row == 1:
            ax.set_xlabel('Month', fontsize=13)

# ── legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=colors['all'], lw=2, label='All $k$'),
    Line2D([0], [0], color=colors['123'], lw=2, label='$k = 1$–$3$'),
    Line2D([0], [0], color=colors['gt3'], lw=2, label='$k > 3$'),
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='6-hourly'),
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='Daily'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=12)

plt.suptitle(f'Effect of cut_ends on EFP annual cycle by spatial scale ({TS.replace("_", "–")})',
             fontsize=14, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])

PLOTS = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/spatial_scale_comparison/plots'
os.makedirs(PLOTS, exist_ok=True)
save_path = os.path.join(PLOTS, 'cut_ends_comparison.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")


# ── Figure 2: 2×3 — rows: frequency | cols: k-group ──────────────────────────
# colours: NH vs SH | linestyles: cut_ends=True/False
hemi_colors = {'nh': cmap.colors[0], 'sh': cmap.colors[1]}

panel_cfg = [
    # (row, col, nh_key, sh_key, panel_label, col_title)
    (0, 0, 'efp_nh',     'efp_sh',     'a', 'All $k$'),
    (0, 1, 'efp_nh_123', 'efp_sh_123', 'b', '$k = 1$–$3$'),
    (0, 2, 'efp_nh_gt3', 'efp_sh_gt3', 'c', '$k > 3$'),
    (1, 0, 'efp_nh',     'efp_sh',     'd', 'All $k$'),
    (1, 1, 'efp_nh_123', 'efp_sh_123', 'e', '$k = 1$–$3$'),
    (1, 2, 'efp_nh_gt3', 'efp_sh_gt3', 'f', '$k > 3$'),
]
freq_rows   = {0: '6h', 1: 'daily'}
freq_labels = {'6h': '6-hourly', 'daily': 'Daily'}

fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8), sharey='row', sharex=True)

for row, col, nh_key, sh_key, panel_label, col_title in panel_cfg:
    ax = axes2[row, col]
    freq = freq_rows[row]

    for cut_label, ls in (('no_cut', '--'), ('cut', '-')):
        d = data[cut_label]['mean'][freq]
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
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='cut_ends=True'),
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='cut_ends=False'),
]
fig2.legend(handles=legend2_handles, loc='lower center', ncol=4,
            bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=12)

plt.suptitle(f'cut_ends comparison by spatial scale and frequency ({TS.replace("_", "–")})',
             fontsize=14, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])

save_path2 = os.path.join(PLOTS, 'cut_ends_comparison_6panel.png')
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path2}")
