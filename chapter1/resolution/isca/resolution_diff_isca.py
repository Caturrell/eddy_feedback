import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import functions.eddy_feedback as ef
import functions.data_wrangling as data
import functions.plotting as plot

#--------------------------------------------------------------------------------------------------

def prepare_datasets(base_path, resolutions):
    datasets = {}
    save_path = base_path / 'wrangled'
    save_path.mkdir(parents=True, exist_ok=True)
    
    for res in resolutions:
        exp = f'HS_{res}_100y-nc'
        save_data = save_path / f'{exp}_annual-mean.nc'

        if not save_data.exists():
            print(f'Processing dataset: {exp}')
            nc_path = base_path / exp
            nc_files = list(nc_path.glob('*.nc'))
            ds = xr.open_mfdataset(nc_files, parallel=True, chunks={'time': 360})
            ds_annual_mean = ds.mean('time')
            ds_annual_mean.to_netcdf(save_data)
        else:
            print(f'Dataset {exp} already exists.')

        datasets[exp] = xr.open_dataset(save_data)
        
    return datasets

#--------------------------------------------------------------------------------------------------

def plot_resolution_comparison(datasets, resolutions, save_dir):
    print('Creating resolution comparison plot...')
    fig, axes = plt.subplots(nrows=len(resolutions), ncols=2, figsize=(8, len(resolutions)*4), sharex=True, sharey=True)

    for i, res in enumerate(resolutions):
        exp = f'HS_{res}_100y-nc'
        datasets[exp].ubar.plot.contourf(levels=20, yincrease=False, ax=axes[i,0])
        datasets[exp].divFy.plot.contourf(levels=20, yincrease=False, ax=axes[i,1])
        for ax in axes[i,:]:
            ax.set_title(f'{res}')

    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / f'HS_resolution_comparison.png')
    print('Resolution comparison plot saved.')

#--------------------------------------------------------------------------------------------------

def compute_and_plot_differences(datasets, ref_res, comp_res_list, save_dir):
    print('Computing resolution differences and plotting...')
    ref_exp = f'HS_{ref_res}_100y-nc'
    fig, axes = plt.subplots(len(comp_res_list), 2, figsize=(10, 4 * len(comp_res_list)), sharex=True, sharey=True)

    for i, res in enumerate(comp_res_list):
        exp = f'HS_{res}_100y-nc'
        interp = datasets[ref_exp].interp(
            lat=datasets[exp].lat, level=datasets[exp].level
        ).sel(lat=slice(-85, 85), level=slice(925., 100.))
        diff = interp - datasets[exp].sel(lat=slice(-85, 85), level=slice(925., 100.))

        diff.ubar.plot.contourf(yincrease=False, ax=axes[i, 0], levels=np.linspace(-32, 32, 17))
        diff.divFy.plot.contourf(yincrease=False, ax=axes[i, 1], levels=np.linspace(-4, 4, 17))

        axes[i, 0].set_title(f'Ubar: {ref_res} - {res}')
        axes[i, 1].set_title(f'divFy: {ref_res} - {res}')
        axes[i, 1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(save_dir / f'HS_{ref_res}_diffs.png')
    print('Resolution difference plots saved.')

#--------------------------------------------------------------------------------------------------

def main():
    base_path = Path('/home/links/ct715/data_storage/isca/held-suarez')
    resolutions = ['T21', 'T42', 'T63', 'T85', 'T170']
    datasets = prepare_datasets(base_path, resolutions)

    script_path = Path(__file__)
    save_dir = script_path.parent / 'plots' / script_path.stem

    plot_resolution_comparison(datasets, ['T21', 'T42', 'T63', 'T85', 'T170'], save_dir)
    compute_and_plot_differences(datasets, 'T85', ['T42', 'T21'], save_dir)

    # Optional: more comparisons
    compute_and_plot_differences(datasets, 'T170', ['T85', 'T63', 'T42'], save_dir)

if __name__ == '__main__':
    main()
