#!/usr/bin/env python3
"""
Plot 2x2 boxplots for EFP bootstrap distributions.

Figure 1 & 2 (freq vs efp_type): one per period (1958_2016, 1979_2016).
  Subplots: rows = efp_full/efp_500, cols = NH(DJF)/SH(JAS).
  Boxes: variables grouped, 6h and daily side-by-side, colored by frequency.

Figure 3 & 4 (efp_type vs freq): one per period.
  Subplots: rows = daily/6h, cols = NH(DJF)/SH(JAS).
  Boxes: variables grouped, efp_full and efp_500 side-by-side, colored by EFP type.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Settings
DATA_DIR = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/data/reanalysis'
SAVE_DIR = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/plots'
EFP_TYPES = ['efp_full', 'efp_500']
PERIODS = ['1958_2016', '1979_2016']
FREQS = ['6h', 'daily']
VARS = ['div1_QG_123', 'div1_QG_gt3', 'div1_QG']
SEASONS = ['djf', 'jas']  # djf for NH, jas for SH

def load_bootstrap_data(efp_type, period, freq, var, season):
    """Load the bootstrap .npy file for given parameters."""
    path = os.path.join(DATA_DIR, efp_type, period, freq, f"{var}_{season}-jra55_efp.npy")
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"Warning: File not found {path}")
        return np.array([])

def plot_freq_comparison(period, var_labels):
    """Figure per period: rows=efp_full/efp_500, cols=NH/SH, boxes compare 6h vs daily per variable."""
    colors = {'6h': 'steelblue', 'daily': 'darkorange'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'EFP Bootstrap - Frequency Comparison - {period}', fontsize=16)

    subplot_configs = [
        ('NH - Full Column', 'efp_full', 'djf'),
        ('SH - Full Column', 'efp_full', 'jas'),
        ('NH - 500 hPa',     'efp_500',  'djf'),
        ('SH - 500 hPa',     'efp_500',  'jas'),
    ]

    for idx, (title, efp_type, season) in enumerate(subplot_configs):
        ax = axes.flat[idx]
        ax.set_title(title)

        data_to_plot, labels, box_colors = [], [], []
        for var in VARS:
            for freq in FREQS:
                boot = load_bootstrap_data(efp_type, period, freq, var, season)
                data_to_plot.append(boot if boot.size > 0 else np.array([np.nan]))
                labels.append(f"{var_labels[var]}\n{freq}")
                box_colors.append(colors[freq])

        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylim(0, 0.6)
        ax.set_ylabel('EFP')
        ax.grid(True, alpha=0.3)

    handles = [plt.Rectangle((0,0),1,1, facecolor=colors[f], alpha=0.7, label=f) for f in FREQS]
    fig.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    return fig


def plot_efptype_comparison(period, var_labels):
    """Figure per period: rows=daily/6h, cols=NH/SH, boxes compare efp_full vs efp_500 per variable."""
    colors = {'efp_full': 'steelblue', 'efp_500': 'darkorange'}
    type_labels = {'efp_full': 'full', 'efp_500': '500hPa'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'EFP Bootstrap - Full vs 500 hPa Comparison - {period}', fontsize=16)

    subplot_configs = [
        ('NH - Daily', 'daily', 'djf'),
        ('SH - Daily', 'daily', 'jas'),
        ('NH - 6-hourly', '6h',  'djf'),
        ('SH - 6-hourly', '6h',  'jas'),
    ]

    for idx, (title, freq, season) in enumerate(subplot_configs):
        ax = axes.flat[idx]
        ax.set_title(title)

        data_to_plot, labels, box_colors = [], [], []
        for var in VARS:
            for efp_type in EFP_TYPES:
                boot = load_bootstrap_data(efp_type, period, freq, var, season)
                data_to_plot.append(boot if boot.size > 0 else np.array([np.nan]))
                labels.append(f"{var_labels[var]}\n{type_labels[efp_type]}")
                box_colors.append(colors[efp_type])

        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylim(0, 0.6)
        ax.set_ylabel('EFP')
        ax.grid(True, alpha=0.3)

    handles = [plt.Rectangle((0,0),1,1, facecolor=colors[t], alpha=0.7, label=type_labels[t]) for t in EFP_TYPES]
    fig.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    return fig


def main():
    var_labels = {'div1_QG_123': 'k1-3', 'div1_QG_gt3': 'k>3', 'div1_QG': 'all-k'}
    os.makedirs(SAVE_DIR, exist_ok=True)

    for period in PERIODS:
        fig1 = plot_freq_comparison(period, var_labels)
        path1 = os.path.join(SAVE_DIR, f'bootstrapped_efp_boxplots_{period}.png')
        fig1.savefig(path1, dpi=150)
        print(f"Saved figure to {path1}")
        plt.close(fig1)

        fig2 = plot_efptype_comparison(period, var_labels)
        path2 = os.path.join(SAVE_DIR, f'bootstrapped_efp_fullvs500_boxplots_{period}.png')
        fig2.savefig(path2, dpi=150)
        print(f"Saved figure to {path2}")
        plt.close(fig2)

if __name__ == "__main__":
    main()
