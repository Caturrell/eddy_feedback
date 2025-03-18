import xarray as xr
from pathlib import Path
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import patches

import functions.eddy_feedback as ef
import functions.data_wrangling as data
import functions.plotting as plot

import pdb

#--------------------------------------------------------------------------------------------------

# IMPORT DATA

# data directories
path = Path('/home/links/ct715/data_storage/isca/held-suarez')
exp_list = ['HS_T21_100y-nc', 'HS_T42_100y-nc', 'HS_T85_100y-nc']

datasets = {}
for exp in exp_list:
    
    # save path
    save_path = path / 'wrangled'
    save_path.mkdir(parents=True, exist_ok=True)
    save_data = save_path / f'{exp}_annual-mean.nc'
    
    if save_data.exists():
        print(f'Dataset {exp} already exists.')
    else:
        print(f'Time mean for {exp} does not exist.')
        nc_path = Path(path) / exp
        nc_files = list(nc_path.glob('*.nc'))
        
        print('Opening dataset...')
        ds = xr.open_mfdataset(
            nc_files,
            parallel=True,
            chunks={'time': 360}
        )

        print(f'Saving data set for: {save_data}')
        ds_annual_mean = ds.mean('time')
        ds_annual_mean.to_netcdf(save_data)
    
    print(f'Opening dataset: {save_data}')
    # Import annual mean data
    ds = xr.open_dataset(save_data)
    datasets[exp] = ds
        
#--------------------------------------------------------------------------------------------------

# RESOLUTION COMPARISON
    
print('Making plot for resolution comparison.')
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(8,14), sharey=True, sharex=True)

datasets['HS_T21_100y-nc'].ubar.plot.contourf(levels=20, yincrease=False, ax=axes[0,0])
datasets['HS_T21_100y-nc'].divFy.plot.contourf(levels=20, yincrease=False, ax=axes[0,1])

datasets['HS_T42_100y-nc'].ubar.plot.contourf(levels=20, yincrease=False, ax=axes[1,0])
datasets['HS_T42_100y-nc'].divFy.plot.contourf(levels=20, yincrease=False, ax=axes[1,1])

datasets['HS_T85_100y-nc'].ubar.plot.contourf(levels=20, yincrease=False, ax=axes[2,0])
datasets['HS_T85_100y-nc'].divFy.plot.contourf(levels=20, yincrease=False, ax=axes[2,1])

for ax in axes[0,:]:
    ax.set_title('T21')
for ax in axes[1,:]:
    ax.set_title('T42')
for ax in axes[2,:]:
    ax.set_title('T85')
    
parent = Path(__file__).parent
file_name = Path(__file__).stem
plot_folder = parent / 'plots' / file_name
plot_folder.mkdir(exist_ok=True, parents=True)
save_plot = plot_folder / f'HS_T21-85_resolution_comparison.png'
plt.savefig(save_plot)

print('Resolution comparison plot complete.')

#--------------------------------------------------------------------------------------------------

# RESOLUTION DIFFERENCE

print('Calculating resoultion differences...')
t85_interp42 = datasets['HS_T85_100y-nc'].interp(lat=datasets['HS_T42_100y-nc'].lat.values)
t85_interp42 = t85_interp42.interp(level=datasets['HS_T42_100y-nc'].level.values)
diff_85_42 = t85_interp42 - datasets['HS_T42_100y-nc']
# # subset data
diff_85_42 = diff_85_42.sel(lat=slice(-85,85))
diff_85_42 = diff_85_42.sel(level=slice(925., 100.))

t85_interp21 = datasets['HS_T85_100y-nc'].interp(lat=datasets['HS_T21_100y-nc'].lat.values)
t85_interp21 = t85_interp21.interp(level=datasets['HS_T21_100y-nc'].level.values)
diff_85_21 = t85_interp21 - datasets['HS_T21_100y-nc']
# # subset data
diff_85_21 = diff_85_21.sel(lat=slice(-85,85))
diff_85_21 = diff_85_21.sel(level=slice(925., 100.))

print('Differences calculated. Now plotting...')

fig, axes = plt.subplots(2,2, sharex=True, sharey=True, figsize=(8,8))

diff_85_21.ubar.plot.contourf(yincrease=False, ax=axes[0,0],
                                levels=np.linspace(-32,32,17))
diff_85_21.divFy.plot.contourf(yincrease=False, ax=axes[0,1],
                                levels=np.linspace(-4,4,17))

diff_85_42.ubar.plot.contourf(yincrease=False, ax=axes[1,0],
                                levels=np.linspace(-32,32,17))
diff_85_42.divFy.plot.contourf(yincrease=False, ax=axes[1,1],
                                levels=np.linspace(-4,4,17))

for ax in axes[0,:]:
    ax.set_title('T85-T21')
    ax.set_xlabel('')
for ax in axes[1,:]:
    ax.set_title('T85-T42')
    
for ax in axes[:,1]:
    ax.set_ylabel('')
    
save_plot2 = plot_folder / f'HS_res_differences.png'
plt.savefig(save_plot2)

#--------------------------------------------------------------------------------------------------

# RESOLUTION DIFFERENCE (JUST T85-T42)

fig, axes = plt.subplots(1,2, sharey=True, figsize=(10,7))

diff_85_42.ubar.plot.contourf(yincrease=False, ax=axes[0],
                                levels=21, cbar_kwargs={'orientation': 'horizontal', 'location': 'bottom'})
diff_85_42.divFy.plot.contourf(yincrease=False, ax=axes[1],
                                levels=np.linspace(-4,4,17), cbar_kwargs={'orientation': 'horizontal', 'location': 'bottom'})

    

axes[1].set_ylabel('')
axes[1].set_title('divFy')
axes[0].set_title('Ubar')
    
save_plot2 = plot_folder / f'HS_T85-T42_res_diff.png'
fig.suptitle('T85-T42')
plt.savefig(save_plot2)