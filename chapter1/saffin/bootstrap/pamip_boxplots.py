import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import logging
import pdb

import functions.data_wrangling as dw
import functions.eddy_feedback as ef


#--------------------------------

# functions

def bootstrap_resampling_pamip(ds, num_samples=1000):
    """
    Perform bootstrap resampling over 'time' dimension of the dataset.

    Parameters:
    ds (xarray.Dataset): The dataset to resample.
    num_samples (int): The number of bootstrap samples to generate.

    Returns:
    list: A list of resampled datasets.
    """
    
    # ds = ds.mean('time')
    
    if 'ens_ax' in ds.dims:
        bootstrap_indices = np.random.choice(ds.sizes["ens_ax"], size=(num_samples, ds.sizes["ens_ax"]), replace=True)
        bootstrap_dssets = [ds.isel(ens_ax=idx) for idx in bootstrap_indices]
    else:
        bootstrap_indices = np.random.choice(ds.sizes["time"], size=(num_samples, ds.sizes["time"]), replace=True)
        bootstrap_dssets = [ds.isel(time=idx) for idx in bootstrap_indices]
    
    return bootstrap_dssets


#--------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

path_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/efp_pd_non-regridded'

# extract model names
files = os.listdir(path_dir)
models = ['CESM1-WACCM-SC']
# models = [os.path.basename(f).split('_')[0] for f in files]
# models.sort()

# models.remove('ECHAM6.3')
# models.remove('E3SMv1')

# Create dictionary containing each model name and dataset
djf_pamip = {}
jas_pamip = {}
for model in models:
    logging.info(f'Loading data for model: {model}')
    # create file path by joining directory and model name
    file_path = os.path.join(path_dir, f'{model}_*.nc')
    # open xarray dataset
    ds = xr.open_mfdataset(file_path, parallel=True, chunks={'time':31})
    
    djf = dw.seasonal_mean(ds, season='djf')
    djf_pamip[model] = djf
    
    jas = dw.seasonal_mean(ds, season='jas')
    jas_pamip[model] = jas



# Initialize a dictionary to store the bootstrap results for each model
bootstrap_results_djf = {}
bootstrap_results_jas = {}


## NH ##

# # Loop through each model in djf_pamip
# for model, ds in djf_pamip.items():
#     logging.info(f'Starting bootstrap resampling for model: {model} (DJF)')
#     # Perform bootstrap resampling
#     bootstrap_djfsets = bootstrap_resampling_pamip(ds)
    
#     # Initialize a list to store the EFP values for each bootstrap sample
#     efp_values = []
    
#     # Loop through each bootstrap sample and calculate EFP values
#     for i, item in enumerate(bootstrap_djfsets):
#         if i % 100 == 0:
#             logging.info(f'Processing bootstrap sample {i+1}/{len(bootstrap_djfsets)} for model: {model} (DJF)')
#         if model == 'CESM1-WACCM-SC':
#             item = item.rename({'time': 'ens_ax'})
#         efp = ef.calculate_efp(item, data_type='pamip', bootstrapping=True)
#         efp_values.append(efp)
        
#     # save the efp values
#     save_path = f'/home/links/ct715/eddy_feedback/chapter1/saffin/bootstrap/data/{model}_djf_efp_values.npy'
#     np.save(save_path, efp_values)
#     logging.info(f'Saved EFP values for model: {model} (DJF) at {save_path}')
    
#     # Store the EFP values in the dictionary
#     bootstrap_results_djf[model] = efp_values
#     logging.info(f'Completed bootstrap resampling for model: {model} (DJF)')

# # Convert bootstrap results to a DataFrame for plotting
# bootstrap_df_djf = pd.DataFrame({
#     'Model': np.repeat(list(bootstrap_results_djf.keys()), [len(v) for v in bootstrap_results_djf.values()]),
#     'EFP': np.concatenate(list(bootstrap_results_djf.values()))
# })

# # Plot the bootstrap results as violin plots using Seaborn
# plt.figure(figsize=(12, 6))
# sns.violinplot(x='Model', y='EFP', data=bootstrap_df_djf)
# plt.xticks(rotation=90)
# plt.title('Bootstrap EFP Values for Each Model (DJF)')
# plt.xlabel('Model')
# plt.ylabel('EFP')
# plt.tight_layout()

# # Save the plot
# plot_path = '/home/links/ct715/eddy_feedback/chapter1/saffin/bootstrap/plots/bootstrap_djf_efp_violinplots.pdf'
# plt.savefig(plot_path)
# logging.info(f'Saved DJF violin plot at {plot_path}')


## SH ##

# Loop through each model in jas_pamip
for model, ds in jas_pamip.items():
    logging.info(f'Starting bootstrap resampling for model: {model} (JAS)')
    # Perform bootstrap resampling
    bootstrap_jassets = bootstrap_resampling_pamip(ds)
    
    # Initialize a list to store the EFP values for each bootstrap sample
    efp_values = []
    
    # Loop through each bootstrap sample and calculate EFP values
    for i, item in enumerate(bootstrap_jassets):
        if i % 100 == 0:
            logging.info(f'Processing bootstrap sample {i+1}/{len(bootstrap_jassets)} for model: {model} (JAS)')
        if model == 'CESM1-WACCM-SC':
            item = item.rename({'time': 'ens_ax'})
        efp = ef.calculate_efp(item, data_type='pamip', calc_south_hemis=True, bootstrapping=True)
        efp_values.append(efp)
        
    # save the efp values
    save_path = f'/home/links/ct715/eddy_feedback/chapter1/saffin/bootstrap/data/{model}_jas_efp_values.npy'
    np.save(save_path, efp_values)
    logging.info(f'Saved EFP values for model: {model} (JAS) at {save_path}')
    
    # Store the EFP values in the dictionary
    bootstrap_results_jas[model] = efp_values
    logging.info(f'Completed bootstrap resampling for model: {model} (JAS)')


bootstrap_df_jas = pd.DataFrame({
    'Model': np.repeat(list(bootstrap_results_jas.keys()), [len(v) for v in bootstrap_results_jas.values()]),
    'EFP': np.concatenate(list(bootstrap_results_jas.values()))
})

# Plot the bootstrap results as violin plots using Seaborn for JAS
plt.figure(figsize=(12, 6))
sns.violinplot(x='Model', y='EFP', data=bootstrap_df_jas)
plt.xticks(rotation=90)
plt.title('Bootstrap EFP Values for Each Model (JAS)')
plt.xlabel('Model')
plt.ylabel('EFP')
plt.tight_layout()

# Save the plot
plot_path = '/home/links/ct715/eddy_feedback/chapter1/saffin/bootstrap/plots/bootstrap_jas_efp_violinplots.pdf'
plt.savefig(plot_path)
logging.info(f'Saved JAS violin plot at {plot_path}')