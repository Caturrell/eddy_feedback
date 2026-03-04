import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from pypalettes import load_cmap

# Load colormap
cmap = load_cmap("highcontrast")

# ============================================================================
# LOAD DATA
# ============================================================================

# Load reanalysis data
obs = pd.read_csv('/home/links/ct715/eddy_feedback/chapter1/reanalysis/data/1979-2016/jra55_efp_k123_1979-2016.csv')

# Extract observed values into nested dictionary
obs_values = {}
for time_freq in ['6h', 'daily']:
    obs_values[time_freq] = {}
    for div1_method in ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']:
        result = obs.query(f"time_freq == '{time_freq}' and div1_method == '{div1_method}'")
        obs_values[time_freq][div1_method] = {
            'nh': result['efp_nh_500'].values[0],
            'sh': result['efp_sh_500'].values[0]
        }

# Load CMIP6 data
path_daily = '/home/links/ct715/eddy_feedback/chapter1/cmip6/data/100y/cmip6_daily_efp_winters_100y.csv'
df_daily = pd.read_csv(path_daily)

path_6h = '/home/links/ct715/eddy_feedback/chapter1/cmip6/data/100y/cmip6_6h_efp_winters_100y.csv'
df_6h = pd.read_csv(path_6h)

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_hemisphere_comparison(ax, df, x_col, y_col, obs_nh, obs_sh, title, time_freq):
    """
    Create a scatter plot comparing NH vs SH EFP with regression line.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    df : pd.DataFrame
        Data to plot
    x_col : str
        Column name for x-axis (SH)
    y_col : str
        Column name for y-axis (NH)
    obs_nh : float
        Observed NH value for reference line
    obs_sh : float
        Observed SH value for reference line
    title : str
        Panel title
    time_freq : str
        Time frequency label for display
    """
    # Regression line (gray background)
    sns.regplot(ax=ax, data=df, x=x_col, y=y_col,
                scatter_kws={'color': 'white'}, 
                line_kws={"color": "gray"}, 
                ci=None)
    
    # Scatter plot with model colors
    sns.scatterplot(ax=ax, data=df, x=x_col, y=y_col,
                    hue='model', style='model', s=150)
    
    # Reference lines from observations
    ax.axhline(y=obs_nh, color='blue', linestyle='--', alpha=0.4)
    ax.axvline(x=obs_sh, color='blue', linestyle='--', alpha=0.4)
    
    # Calculate and display correlation
    r, p = sp.stats.pearsonr(x=df[y_col], y=df[x_col])
    ax.text(0.05, 0.9, f"r = {r:.2f}, p = {p:.3f}", 
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7), 
            fontsize=12)
    
    # Formatting
    ax.set_xlabel('SH EFP', fontsize=14)
    ax.set_ylabel('NH EFP', fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_title(title, fontsize=14)
    ax.legend_.remove()
    ax.grid(visible=True, color='gainsboro')
    ax.set_axisbelow(True)

# ============================================================================
# CREATE FIGURE
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

# Define panel configurations
panels = [
    # Top row: 6-hourly data
    {'row': 0, 'col': 0, 'df': df_6h, 'x': 'efp_sh', 'y': 'efp_nh', 
     'method': 'div1_QG', 'label': '(a)', 'k_label': 'All k'},
    {'row': 0, 'col': 1, 'df': df_6h, 'x': 'efp_sh_123', 'y': 'efp_nh_123', 
     'method': 'div1_QG_123', 'label': '(b)', 'k_label': 'k123'},
    {'row': 0, 'col': 2, 'df': df_6h, 'x': 'efp_sh_gt3', 'y': 'efp_nh_gt3', 
     'method': 'div1_QG_gt3', 'label': '(c)', 'k_label': 'k>3'},
    
    # Bottom row: daily data
    {'row': 1, 'col': 0, 'df': df_daily, 'x': 'efp_sh', 'y': 'efp_nh', 
     'method': 'div1_QG', 'label': '(d)', 'k_label': 'All k'},
    {'row': 1, 'col': 1, 'df': df_daily, 'x': 'efp_sh_123', 'y': 'efp_nh_123', 
     'method': 'div1_QG_123', 'label': '(e)', 'k_label': 'k123'},
    {'row': 1, 'col': 2, 'df': df_daily, 'x': 'efp_sh_gt3', 'y': 'efp_nh_gt3', 
     'method': 'div1_QG_gt3', 'label': '(f)', 'k_label': 'k>3'},
]

# Plot all panels
for panel in panels:
    time_freq = '6h' if panel['row'] == 0 else 'daily'
    title = f"$\\bf{{{panel['label']}}}$ {panel['k_label']}"
    
    plot_hemisphere_comparison(
        ax=axes[panel['row'], panel['col']],
        df=panel['df'],
        x_col=panel['x'],
        y_col=panel['y'],
        obs_nh=obs_values[time_freq][panel['method']]['nh'],
        obs_sh=obs_values[time_freq][panel['method']]['sh'],
        title=title,
        time_freq=time_freq
    )

# Add row labels
fig.text(0.02, 0.75, '6-hourly', va='center', rotation='vertical', 
         fontsize=16, fontweight='bold')
fig.text(0.02, 0.25, 'Daily', va='center', rotation='vertical', 
         fontsize=16, fontweight='bold')

# Add legend to the rightmost panel
axes[0, 2].legend(
    bbox_to_anchor=(1.02, 1.05),
    loc='upper left',
    prop={'size': 8},
    handlelength=1.5,
    handletextpad=0.5
)

plt.tight_layout(rect=[0.03, 0, 1, 0.96])
plt.suptitle('NH vs SH EFP for CMIP6 piControl runs (100 years)', 
             fontsize=16, y=0.98)

# ============================================================================
# SAVE FIGURE
# ============================================================================

save_path = '/home/links/ct715/eddy_feedback/chapter1/cmip6/plots'
filename = 'NHvsSH_6h-daily_1979-2014_all-k_hist.png'
full_path = os.path.join(save_path, filename)

user_input = input(f"Do you want to save the plot to {full_path}? (y/n): ").strip().lower()

if user_input == 'y':
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {full_path}")
else:
    print("Plot not saved.")

plt.show()