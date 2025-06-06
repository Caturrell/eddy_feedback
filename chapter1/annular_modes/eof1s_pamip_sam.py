import logging
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from eofs.xarray import Eof
from pathlib import Path

import functions.eddy_feedback as ef

import pdb

# -------------------------------
# Configure logging
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# 1. Import the datasets into a dictionary
# -------------------------------
dir_path = '/home/links/ct715/data_storage/PAMIP/processed_monthly/combined_ua_epfy_divFy/1.1_pdSST-pdSIC'
logging.info(f"Loading datasets from directory: {dir_path}")
path_dir = Path(dir_path)
models_dir = sorted(path_dir.iterdir())
models = [model.name.split('_')[0] for model in models_dir]

ds = {}
for model in models:
    file_path = f'{dir_path}/{model}_1.1_u_ubar_epfy_divFy.nc'
    logging.info(f"Loading dataset for model: {model} from {file_path}")
    ds[model] = xr.open_dataset(file_path)
    ds[model].attrs['model'] = model
logging.info("All datasets loaded.\n")


# -------------------------------
# 2. Process each model to compute EFP
# -------------------------------
# Create dictionary containing each model name and dataset
logging.info("Calculating EFP for each model...")
pamip_efp_sh = {}
for model in models:
    # Add dataset to dictionary with required model name
    efp_sh = ef.calculate_efp(ds[model], data_type='pamip', calc_south_hemis=True)
    pamip_efp_sh[model] = round(float(efp_sh), 2)
    logging.info(f"EFP for {model}: {pamip_efp_sh[model]}")
logging.info("EFP calculation completed for all models.\n")

# -------------------------------
# 3. Process each model to compute EOF1
# -------------------------------
eof_dict = {}
variance_fractions = {}
logging.info("Starting EOF analysis for each model...")
for model, ds_model in ds.items():
    logging.info(f"Processing model: {model}")
    # Select u at 850 hPa and resample to monthly means
    u = ds_model['u'].sel(level=slice(1000,100))
    u = u.mean(('ens_ax', 'level'))
    u_mon = u.resample(time='1ME').mean()
    
    # Remove the time mean to compute anomalies
    u_mon_anom = u_mon - u_mon.mean(dim='time')
    
    # Ensure that the 'time' dimension is the first dimension
    # This is required by the Eof solver.
    u_mon_anom = u_mon_anom.transpose("time", "lat", "lon")
    
    # Perform EOF analysis (expects dimensions: time, lat, lon)
    solver_mon = Eof(u_mon_anom)
    
    # Retrieve the first EOF (removing the extra 'mode' dimension)
    eof1_mon = solver_mon.eofsAsCovariance(neofs=1).isel(mode=0)
    variance_fractions[model] = solver_mon.varianceFraction(neigs=1)
    
    eof_dict[model] = eof1_mon
    logging.info(f"EOF1 computed for model: {model}")
logging.info("EOF analysis completed for all models.\n")

# pdb.set_trace()

# -------------------------------
# 4. Plot all EOF1's on a single figure
# -------------------------------
logging.info("Starting plotting of EOF1 fields for each model...")
n_models = len(eof_dict)
n_cols = int(np.ceil(np.sqrt(n_models)))
n_rows = int(np.ceil(n_models / n_cols))
logging.info(f"Creating a grid of {n_rows} rows and {n_cols} columns for {n_models} models.")

# Define the polar projection centered on the South Pole.
proj = ccrs.SouthPolarStereo()
fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection': proj}, figsize=(12, 12))
axes = np.atleast_1d(axes).flatten()  # ensure axes is a 1D array

# Loop over each model and plot EOF1
for ax, (model, eof1) in zip(axes, eof_dict.items()):
    logging.info(f"Plotting EOF1 for model: {model}")
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    levels=np.linspace(-0.035, 0.035, 21)
    contour = ax.contourf(eof1.lon, eof1.lat, eof1,
                          levels=np.linspace(-12,12, 21),
                          transform=ccrs.PlateCarree(),
                          cmap='PuOr_r')
    ax.coastlines()
    ax.gridlines()
    ax.set_title(f"EOF1: {round(variance_fractions[model].values[0],2)*100}% - {model} ({pamip_efp_sh[model]})")

# Remove any unused subplots if grid is larger than number of models
for ax in axes[len(eof_dict):]:
    ax.remove()

# Add a common horizontal colorbar
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
plt.suptitle("EOF1 of Zonal Wind at 850 hPa - PAMIP Models", fontsize=16)

# Save and display the plot
plot_path = f'/home/links/ct715/eddy_feedback/chapter1/annular_modes/plots/PAMIP_monthly_EOF1_AsCov.png'
plt.savefig(plot_path)
logging.info(f"Figure saved to {plot_path}")
plt.show()
logging.info("Plotting completed.")
