"""
python /home/links/ct715/eddy_feedback/EPflux/nakamura_plot.py
"""


import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
sys.path.append('/home/links/ct715/eddy_feedback')
from functions.aos_functions import PlotEPfluxArrows

# pull in dataset
ds = xr.open_mfdataset('/home/links/ct715/eddy_feedback/daily_datasets/jra55_djf_ep.nc')

#---------------------------------------------------------------------------------------------------------

# Skip datapoints for better visualisation

# skip both
skip_lat = 3
skip_pres = 1
skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )

# define variables
x = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
y = ds.level.isel(dict(level=slice(None, None, skip_pres)))
u = ds.ep1.mean(('time')).isel(skip)
v = ds.ep2.mean(('time')).isel(skip)

#---------------------------------------------------------------------------------------------------------

# Create plot

# choose colour palette
coolwarm = sns.color_palette("coolwarm", as_cmap=True)

# start figure
fig, ax = plt.subplots(figsize=(9,5))

# plot colour map
plt.contourf(ds.lat.values, ds.level.values, ds.ubar, 
             cmap=coolwarm, levels=np.linspace(-60,60,15))
plt.colorbar(location='bottom', orientation='horizontal', shrink=0.5, 
             ticks=np.arange(-60,70,20), label='Wind speed (m/s)')

# Figure settings
plt.yscale('log')
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')
plt.title('Winter Months (DJF)')


# plot EP flux
PlotEPfluxArrows(x, y, u, v,
                fig, ax, pivot='mid', yscale='log')

plt.savefig('/home/links/ct715/eddy_feedback/EPflux/nakamura_jra55.png')
plt.show()