import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import functions.data_wrangling as dw

# ============================================================================
# SETUP
# ============================================================================
# Load data
path = '/home/links/ct715/data_storage/reanalysis/jra55_daily/processed_efp'
path_6h = os.path.join(path, 'k123_6h_ubar_epf-pr-QG_1MS_1958-2016.nc')
ds_6h = xr.open_dataset(path_6h)

# Define seasons and variable
seasons = ['djf', 'jfm', 'fma', 'mam', 'amj', 'mjj', 
           'jja', 'jas', 'aso', 'son', 'ond', 'ndj']
corr_var = 'div1_QG_123'

# Define hemispheres
hemispheres = {
    'NH': {'lat_slice': slice(0, 90), 'box_lat': (25, 50)},
    'SH': {'lat_slice': slice(-90, 0), 'box_lat': (-75, 50)}
}

# Output path
save_path = '/home/links/ct715/eddy_feedback/chapter1/reanalysis/plots'

# ============================================================================
# PLOT FOR EACH HEMISPHERE
# ============================================================================
for hemi_name, hemi_config in hemispheres.items():
    print(f'\n{"="*60}')
    print(f'Processing {hemi_name}')
    print(f'{"="*60}')
    
    # Select hemisphere
    ds_hemi = ds_6h.sel(lat=hemi_config['lat_slice'])
    
    # Create figure with 4 rows x 3 columns (12 subplots total)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 16), 
                             sharex=True, sharey=True)
    
    # Loop over seasons
    for idx, season in enumerate(seasons):
        print(f'  Season: {season.upper()}')
        
        # Calculate row and column indices
        row = idx // 3  # Integer division for row
        col = idx % 3   # Modulo for column
        
        # Calculate seasonal mean
        ds_season = dw.seasonal_mean(ds_hemi, season=season, cut_ends=True)
        
        ax = axes[row, col]
        
        # Calculate correlation
        corr = xr.corr(ds_season['ubar'], ds_season[corr_var], dim='time')
        
        # Plot
        im = corr.plot.contourf(ax=ax, vmin=-1, vmax=1, cmap='RdBu_r',
                                 yincrease=False, levels=20, 
                                 add_colorbar=False)
        
        # Add season label as title for each subplot
        ax.set_title(f'{season.upper()}', fontsize=11, fontweight='bold')
        
        # Add y-label only for first column
        if col == 0:
            ax.set_ylabel('Pressure (hPa)', fontsize=10)
        else:
            ax.set_ylabel('')
        
        # Add box for 200-600 hPa, 25-75°N/S
        box = Rectangle((hemi_config['box_lat'][0], 200), 
                        width=hemi_config['box_lat'][1], 
                        height=400, 
                        fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        # Set x-label only for bottom row
        if row == 3:
            ax.set_xlabel('Latitude (°)', fontsize=10)
        else:
            ax.set_xlabel('')
    
    # Adjust spacing
    fig.subplots_adjust(bottom=0.08, top=0.95, hspace=0.25, wspace=0.15)
    
    # Add horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Correlation', fontsize=12, fontweight='bold')
    
    # Add overall title
    fig.suptitle(f'{hemi_name}: Seasonal Correlation Maps (ubar vs. k=1-3)', 
                 fontsize=16, fontweight='bold')
    
    # Save figure
    filename = f'corr_maps_all_seasons_k123_{hemi_name}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    print(f'  Saved: {filename}')
    plt.close()

print(f'\n{"="*60}')
print('Complete!')
print(f'{"="*60}')