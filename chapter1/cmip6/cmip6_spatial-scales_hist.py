import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypalettes import load_cmap

# ── Constants ─────────────────────────────────────────────────────────────────

BOOTSTRAP_DIR = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/data/reanalysis/1979_2016'

EFP_COLS    = ["efp_nh", "efp_nh_gt3", "efp_nh_123", "efp_sh", "efp_sh_gt3", "efp_sh_123"]
DATASET_MAP = {None: 'all k', '_gt3': 'k>3', '_123': 'k123'}

POSITIONS = {
    ("SH", "total"): -0.265, ("SH", "k>3"):  0.0,   ("SH", "k123"):  0.265,
    ("NH", "total"):  0.735, ("NH", "k>3"):  1.0,   ("NH", "k123"):  1.265,
}
HEMI_MAP = {"NH": "nh", "SH": "sh"}

# ── Loop over time frequencies ────────────────────────────────────────────────

for time_freq in ['6h', 'daily']:

    print(f"\n{'─' * 60}\nProcessing: {time_freq}\n{'─' * 60}")

    data_path   = f'/home/links/ct715/eddy_feedback/chapter1/cmip6/data/100y/cmip6_{time_freq}_efp_winters_100y.csv'
    output_path = f'/home/links/ct715/eddy_feedback/chapter1/cmip6/plots/cmip6_piControl_{time_freq}_spatial-scales_100y.png'

    # ── Load and reshape CMIP6 data ───────────────────────────────────────────

    df = pd.read_csv(data_path)

    df_long = df.melt(id_vars="model", value_vars=EFP_COLS,
                      var_name="hemisphere_dataset", value_name="efp")
    df_long[['hemisphere', 'dataset']] = df_long['hemisphere_dataset'].str.extract(r'efp_(nh|sh)(_.*)?')
    df_long['dataset'] = df_long['dataset'].replace(DATASET_MAP)

    # ── Load bootstrap reanalysis data ────────────────────────────────────────

    records = []
    for fname in os.listdir(BOOTSTRAP_DIR):
        parts = fname.split('_')

        # Only load files matching the current time frequency
        if parts[1] != time_freq:
            continue

        data    = np.load(os.path.join(BOOTSTRAP_DIR, fname))
        hemis   = 'nh' if parts[-2] == 'djf-jra55' else 'sh'
        which_k = parts[4]

        label_map = {'123': f'EFP_{hemis}_k123', 'gt3': f'EFP_{hemis}_k>3'}
        column = label_map.get(which_k, f'EFP_{hemis}_total')

        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        records.append({'efp_type': column, 'efp_mean': mean, 'efp_std': std})

        print(f"{fname}\n  {column}: {mean:.3f}  [{mean - std:.3f}, {mean + std:.3f}]")

    bootstrap_df = pd.DataFrame(records)

    # ── Plot ──────────────────────────────────────────────────────────────────

    palette = load_cmap("highcontrast").colors

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))

    plot_kwargs = dict(data=df_long, x='hemisphere', y='efp', hue='dataset',
                       palette=palette, order=['sh', 'nh'])

    sns.boxplot(**plot_kwargs, linewidth=1.2, showfliers=False, ax=ax)
    sns.stripplot(**plot_kwargs, dodge=True, alpha=0.9, size=4, jitter=True,
                  marker="o", linewidth=0.5, edgecolor='k', ax=ax)

    # ── Overlay reanalysis points with error bars ─────────────────────────────

    for (hemi, dataset), x_pos in POSITIONS.items():
        label = f'EFP_{HEMI_MAP[hemi]}_{dataset}'
        row   = bootstrap_df[bootstrap_df['efp_type'] == label].iloc[0]

        ax.errorbar(x_pos, row.efp_mean, yerr=row.efp_std,
                    color='red', capsize=5, capthick=1.5, elinewidth=1.5, zorder=9)
        ax.scatter(x_pos, row.efp_mean,
                   color='red', s=90, edgecolor='k', zorder=10, marker='D')

    # ── Labels, legend, and save ──────────────────────────────────────────────

    ax.set_xlabel("")
    ax.set_ylabel("EFP Value", fontsize=14)
    ax.set_title(f"piControl Winter EFP Spread ({time_freq})", fontsize=16, pad=15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Southern Hemisphere", "Northern Hemisphere"])

    handles, labels = ax.get_legend_handles_labels()
    unique_handles   = handles[:3]
    unique_labels    = labels[:3]
    unique_handles.append(ax.scatter([], [], color='red', edgecolor='k', marker='D', s=90))
    unique_labels.append('Reanalysis ± 1σ')

    ax.legend(unique_handles, unique_labels,
              title="Input data to calculate EP fluxes",
              loc="lower center", bbox_to_anchor=(0.5, -0.25),
              ncol=4, fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")