import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypalettes import load_cmap

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/various_efp_methods/data'
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
    'pr':     cmap.colors[0],
    'QG':     cmap.colors[1],
    'QG_500': cmap.colors[2],
}
styles = {'6h': '-', 'daily': '--'}

# rows: NH (top), SH (bottom) | cols: no_cut (left), cut (right)
row_cfg = [
    ('Northern Hemisphere', 'efp_pr_nh', 'efp_QG_nh'),
    ('Southern Hemisphere', 'efp_pr_sh', 'efp_QG_sh'),
]
col_cfg = [
    ('cut_ends=False', 'no_cut', 'a', 'b'),
    ('cut_ends=True',  'cut',    'c', 'd'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row', sharex=True)

for row, (hemi, pr_key, qg_key) in enumerate(row_cfg):
    for col, (cut_title, cut_label, label_t, label_b) in enumerate(col_cfg):
        ax = axes[row, col]
        panel_label = label_t if row == 0 else label_b
        cut_data = data[cut_label]

        for freq in ('6h', 'daily'):
            ls = styles[freq]

            # EFP_pr
            xs, ys = extract_series(cut_data['mean'][freq], pr_key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=colors['pr'], linestyle=ls)

            # EFP_QG (mean-level)
            xs, ys = extract_series(cut_data['mean'][freq], qg_key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=colors['QG'], linestyle=ls)

            # EFP_QG (500 hPa)
            xs, ys = extract_series(cut_data['500'][freq], qg_key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=colors['QG_500'], linestyle=ls)

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
    Line2D([0], [0], color=colors['pr'],     lw=2, label='EFP$_{\\mathrm{PR}}$'),
    Line2D([0], [0], color=colors['QG'],     lw=2, label='EFP$_{\\mathrm{QG}}$'),
    Line2D([0], [0], color=colors['QG_500'], lw=2, label='EFP$_{\\mathrm{QG}}$ (500 hPa)'),
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='6-hourly'),
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='Daily'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=12)

plt.suptitle(f'Effect of cut_ends on EFP annual cycle ({TS.replace("_", "–")})',
             fontsize=14, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])

save_path = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/various_efp_methods/plots/cut_ends_comparison.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")


# ── Figure 2: 6-panel, one panel per line ─────────────────────────────────────
# rows: 6h (top), daily (bottom) | cols: EFP_pr, EFP_QG, EFP_QG_500
# colours: NH vs SH | solid: cut_ends=True | dashed: cut_ends=False

hemi_colors = {'nh': cmap.colors[0], 'sh': cmap.colors[1]}

panel_cfg = [
    # (row, col, nh_key, sh_key, level_label, panel_label)
    (0, 0, 'efp_pr_nh',  'efp_pr_sh',  'mean', 'PR',          'a'),
    (0, 1, 'efp_QG_nh',  'efp_QG_sh',  'mean', 'QG',          'b'),
    (0, 2, 'efp_QG_nh',  'efp_QG_sh',  '500',  'QG (500 hPa)','c'),
    (1, 0, 'efp_pr_nh',  'efp_pr_sh',  'mean', 'PR',          'd'),
    (1, 1, 'efp_QG_nh',  'efp_QG_sh',  'mean', 'QG',          'e'),
    (1, 2, 'efp_QG_nh',  'efp_QG_sh',  '500',  'QG (500 hPa)','f'),
]
freq_rows = {0: '6h', 1: 'daily'}
freq_labels = {'6h': '6-hourly', 'daily': 'Daily'}

fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8), sharey='row', sharex=True)

for row, col, nh_key, sh_key, level, efp_label, panel_label in panel_cfg:
    ax = axes2[row, col]
    freq = freq_rows[row]

    for cut_label, ls in (('no_cut', '--'), ('cut', '-')):
        d = data[cut_label][level][freq]
        for hemi, key in (('nh', nh_key), ('sh', sh_key)):
            xs, ys = extract_series(d, key)
            ax.plot(xs, ys, marker='o', linewidth=2,
                    color=hemi_colors[hemi], linestyle=ls)

    ax.set_xticks(range(12))
    ax.set_xticklabels([MONTH_MAP[m] for m in MONTH_ORDER])
    ax.grid(True, axis='y', alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.set_title(
        f'$\\bf{{({panel_label})}}$ {freq_labels[freq]} — EFP$_{{\\mathrm{{{efp_label}}}}}$',
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

plt.suptitle(f'cut_ends comparison by EFP type and frequency ({TS.replace("_", "–")})',
             fontsize=14, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])

save_path2 = '/home/links/ct715/eddy_feedback/chapter1/annual_cycle/various_efp_methods/plots/cut_ends_comparison_6panel.png'
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path2}")
