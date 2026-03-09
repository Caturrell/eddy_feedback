import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import pearsonr

# Setup
base_dir = '/gws/ssde/j25a/arctic_connect/cturrell/CMIP6/historical'
model_list = os.listdir(base_dir)
model_list.remove('EC-Earth3-Veg-LR')
model_list.remove('IPSL-CM6A-LR')
model_list.remove('IPSL-CM6A-LR-INCA')
model_list.remove('KACE-1-0-G')

# Load EFP data once
efp_df = pd.read_csv('/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/data/1979_2014/cmip6_hist_1979_2014_6h_efp_winters.csv')

# Collect data for all models
model_data = []

for model in model_list:
    try:
        # Navigate to model directory
        if len(os.listdir(os.path.join(base_dir, model))) > 1:
            print(f'Warning: {model} has more than one time period, skipping')
            continue
        
        model_dir = os.path.join(base_dir, model, os.listdir(os.path.join(base_dir, model))[0])
        data_dir = os.path.join(model_dir, '6hrPlevPt', 'yearly_data')
        
        # Get longitude resolution
        files = [f for f in os.listdir(data_dir) if f.endswith('_daily_averages.nc')]
        if not files:
            print(f'Warning: No files found for {model}, skipping')
            continue
            
        first_file = files[0]
        ds = xr.open_dataset(os.path.join(data_dir, first_file))
        lon_res = ds['lon'].diff('lon').values[0]
        ds.close()
        
        # Get EFP values
        model_efp = efp_df[efp_df['model'] == model]
        if model_efp.empty:
            print(f'Warning: No EFP data found for {model}, skipping')
            continue
            
        efp_nh = model_efp['efp_nh'].values[0]
        efp_sh = model_efp['efp_sh'].values[0]
        
        # Store results
        model_data.append({
            'model': model,
            'lon_res': lon_res,
            'efp_nh': efp_nh,
            'efp_sh': efp_sh
        })
        
        print(f'Processed {model}: lon_res={lon_res:.2f}, EFP_NH={efp_nh:.3f}, EFP_SH={efp_sh:.3f}')
        
    except Exception as e:
        print(f'Error processing {model}: {e}')
        continue

# Convert to DataFrame for easier plotting
results_df = pd.DataFrame(model_data)

# Calculate Pearson correlations
nh_corr, nh_pval = pearsonr(results_df['lon_res'], results_df['efp_nh'])
sh_corr, sh_pval = pearsonr(results_df['lon_res'], results_df['efp_sh'])

# Create the two-column plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Northern Hemisphere (left panel)
sns.regplot(x='lon_res', y='efp_nh', data=results_df, 
            ax=axes[0], 
            scatter_kws={'s': 100, 'alpha': 0.6}, 
            color='#d62728',
            line_kws={'linewidth': 2})
axes[0].set_xlabel('Longitude Resolution (degrees)', fontsize=12)
axes[0].set_ylabel('EFP (days$^{-1}$)', fontsize=12)
axes[0].set_title('Northern Hemisphere', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Add correlation text to NH panel
axes[0].text(0.05, 0.95, f'r = {nh_corr:.3f}\np = {nh_pval:.3f}',
             transform=axes[0].transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Southern Hemisphere (right panel)
sns.regplot(x='lon_res', y='efp_sh', data=results_df, 
            ax=axes[1], 
            scatter_kws={'s': 100, 'alpha': 0.6}, 
            color='#1f77b4',
            line_kws={'linewidth': 2})
axes[1].set_xlabel('Longitude Resolution (degrees)', fontsize=12)
axes[1].set_title('Southern Hemisphere', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')

# Add correlation text to SH panel
axes[1].text(0.05, 0.95, f'r = {sh_corr:.3f}\np = {sh_pval:.3f}',
             transform=axes[1].transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Overall title
fig.suptitle('Eddy Feedback Parameter vs Longitude Resolution\nCMIP6 Historical (1979-2014, DJF)', 
             fontsize=14, fontweight='bold', y=1.00)

plt.tight_layout()
save_path = '/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/analysis_historical/efp_vs_lonres_hist.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f'\nProcessed {len(results_df)} models successfully')
print(f'\nSummary statistics:')
print(results_df[['lon_res', 'efp_nh', 'efp_sh']].describe())
print(f'\nNH correlation: r={nh_corr:.3f}, p={nh_pval:.3f}')
print(f'SH correlation: r={sh_corr:.3f}, p={sh_pval:.3f}')