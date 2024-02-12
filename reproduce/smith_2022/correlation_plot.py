"""
python /home/links/ct715/eddy_feedback/correlation_plot.py
"""


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sys 
sys.path.append('/home/links/ct715/eddy_feedback')
import functions.eddy_feedback as ef

import pdb 

# import dataset on MATHS SERVER
# ds = xr.open_mfdataset('/home/links/ct715/eddy_feedback/daily_datasets/jra55_djf_ep.nc')

# Import dataset on JASMIN
ds = xr.open_mfdataset('/home/links/ct715/eddy_feedback/daily_datasets/jra55_uvtw_ep.nc',
                            parallel=True, chunks={'time': 31})

# NH subset
# ds = ds.isel(lat=slice(0, 37))

#------------------------------------------------------------------------------------------

# define variables as same shape
ubar = ds.u.mean(('lon'))
div1 = ds.div1
div2 = ds.div2

# separate time into annual means
# and use .load() to force the calculation now
ubar = ubar.groupby('time.year').mean('time').load()
div1 = div1.groupby('time.year').mean('time').load()
div2 = div2.groupby('time.year').mean('time').load()

pdb.set_trace()

corr = ef.correlation_array(ubar, div1)

#------------------------------------------------------------------------------------------

# choose colour palette
coolwarm = sns.color_palette("coolwarm", as_cmap=True)

# plot figure
plt.figure()

plt.contourf(ds.lat.values, ds.level.values, corr, cmap=coolwarm, levels=10)
plt.colorbar(location='bottom', orientation='horizontal', shrink=0.75,
             label='correlation', extend='both')
# plt.yscale('log')
plt.gca().invert_yaxis()

plt.xlabel('Latitude $(^\\circ N)$')
plt.ylabel('Log pressure (hPa)')
plt.title('(a) $Corr(\\bar{u}, \\nabla_{\\phi} F_{\\phi})$')

plt.savefig('/home/links/ct715/eddy_feedback/corr_linear_whole.png')
plt.show()