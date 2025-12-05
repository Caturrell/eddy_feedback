import xarray as xr
import matplotlib.pyplot as plt
import os
import pdb

import functions.eddy_feedback as ef

data_path = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit/30y/mon_avg_daily'
file_list = [file for file in os.listdir(data_path) if file.endswith('.nc')]

datasets = {}
for file in file_list:
    file_path = os.path.join(data_path, file)
    model_name = file.split('_')[0]
    
    ds = xr.open_dataset(file_path, use_cftime=True)
    datasets[model_name] = ds
    
save_path = '/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/plots/check_mon_avg_data'

for model, ds in datasets.items():
    
    efp_nh = ef.calculate_efp(ds, )
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 10))
    
    ds.ubar.mean('time').plot.contourf(ax=ax1, levels=21)
    ax1.set_title(f'{model} - Ubar')
    
    ds.div1_QG.mean('time').plot.contourf(ax=ax2, levels=21)
    ax2.set_title(f'{model} - div1_QG')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model}_ubar_div1-QG.png'))
    plt.close()
    
    