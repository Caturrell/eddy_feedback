import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypalettes import load_cmap

# ── Config ────────────────────────────────────────────────────────────────────

COMBINED_FIGURE = True   # True = one figure with 2 cols; False = 2 separate figures

BOOTSTRAP_DIR = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/data/reanalysis/1979_2016'

EFP_COLS      = ["efp_nh", "efp_nh_gt3", "efp_nh_123", "efp_sh", "efp_sh_gt3", "efp_sh_123"]
DATASET_MAP   = {None: 'all k', '_gt3': 'k>3', '_123': 'k123'}
HEMI_ORDER    = ['sh', 'nh']   # controls left-to-right order on x-axis
HEMI_LABELS   = ["Southern Hemisphere", "Northern Hemisphere"]

POSITIONS = {
    ("SH", "total"): -0.265, ("SH", "k>3"):  0.0,   ("SH", "k123"):  0.265,
    ("NH", "total"):  0.735, ("NH", "k>3"):  1.0,   ("NH", "k123"):  1.265,
}
HEMI_MAP = {"NH": "nh", "SH": "sh"}

TIME_FREQS = ['6h', 'daily']

# ── Helper: load data for one time frequency ──────────────────────────────────

def load_cmip6(time_freq):
    path = f'/home/links/ct715/eddy_feedback/chapter1/cmip6/data/100y/cmip6_{time_freq}_efp_winters_100y.csv'
    df = pd.read_csv(path)
    df_long = df.melt(id_vars="model", value_vars=EFP_COLS,
                      var_name="hemisphere_dataset", value_name="efp")
    df_long[['hemisphere', 'dataset']] = df_long['hemisphere_dataset'].str.extract(r'efp_(nh|sh)(_.*)?')
    df_long['dataset'] = df_long['dataset'].replace(DATASET_MAP)
    return df_long


def load_bootstrap(time_freq):
    records = []
    for fname in os.listdir(BOOTSTRAP_DIR):
        parts = fname.split('_')
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
    return pd.DataFrame(records)


# ── Helper: populate a single axes ───────────────────────────────────────────

def plot_efp(ax, df_long, bootstrap_df, palette, show_ylabel=True):
    plot_kwargs = dict(data=df_long, x='hemisphere', y='efp', hue='dataset',
                       order=HEMI_ORDER, palette=palette)

    sns.boxplot(**plot_kwargs, linewidth=1.2, showfliers=False, ax=ax)
    sns.stripplot(**plot_kwargs, dodge=True, alpha=0.9, size=4, jitter=True,
                  marker="o", linewidth=0.5, edgecolor='k', ax=ax)

    for (hemi, dataset), x_pos in POSITIONS.items():
        label = f'EFP_{HEMI_MAP[hemi]}_{dataset}'
        row   = bootstrap_df[bootstrap_df['efp_type'] == label].iloc[0]
        ax.errorbar(x_pos, row.efp_mean, yerr=row.efp_std,
                    color='red', capsize=5, capthick=1.5, elinewidth=1.5, zorder=9)
        ax.scatter(x_pos, row.efp_mean,
                   color='red', s=90, edgecolor='k', zorder=10, marker='D')

    ax.set_xlabel("")
    ax.set_ylabel("EFP Value" if show_ylabel else "", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(HEMI_LABELS)
    ax.legend().remove()


def make_legend(ax, palette):
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = handles[:3]
    unique_labels  = labels[:3]
    unique_handles.append(ax.scatter([], [], color='red', edgecolor='k', marker='D', s=90))
    unique_labels.append('Reanalysis ± 1σ')
    return unique_handles, unique_labels


# ── Main ──────────────────────────────────────────────────────────────────────

palette = load_cmap("highcontrast").colors
sns.set_theme(style="whitegrid", context="talk")

if COMBINED_FIGURE:
    output_path = '/home/links/ct715/eddy_feedback/chapter1/cmip6/plots/cmip6_piControl_6h+daily_spatial-scales_100y.png'

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    for ax, time_freq in zip(axes, TIME_FREQS):
        print(f"\n{'─' * 60}\nProcessing: {time_freq}\n{'─' * 60}")
        df_long      = load_cmip6(time_freq)
        bootstrap_df = load_bootstrap(time_freq)
        show_ylabel  = (time_freq == TIME_FREQS[0])
        plot_efp(ax, df_long, bootstrap_df, palette, show_ylabel=show_ylabel)
        ax.set_title(f"piControl Winter EFP Spread ({time_freq})", fontsize=16, pad=15)

    handles, labels = make_legend(axes[0], palette)
    fig.legend(handles, labels,
               title="Input data to calculate EP fluxes",
               loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=4, fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")

else:
    for time_freq in TIME_FREQS:
        print(f"\n{'─' * 60}\nProcessing: {time_freq}\n{'─' * 60}")
        output_path  = f'/home/links/ct715/eddy_feedback/chapter1/cmip6/plots/cmip6_piControl_{time_freq}_spatial-scales_100y.png'
        df_long      = load_cmip6(time_freq)
        bootstrap_df = load_bootstrap(time_freq)

        fig, ax = plt.subplots(figsize=(11, 6))
        plot_efp(ax, df_long, bootstrap_df, palette, show_ylabel=True)
        ax.set_title(f"piControl Winter EFP Spread ({time_freq})", fontsize=16, pad=15)

        handles, labels = make_legend(ax, palette)
        ax.legend(handles, labels,
                  title="Input data to calculate EP fluxes",
                  loc="lower center", bbox_to_anchor=(0.5, -0.25),
                  ncol=4, fontsize=10, title_fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"\nFigure saved to {output_path}")