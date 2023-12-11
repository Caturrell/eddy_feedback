"""
python /home/links/ct715/reanalysis_data/eddy_feedback/correlation_plot.py
"""


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import pdb

# import dataset
ds = xr.open_mfdataset('/home/links/ct715/reanalysis_data/eddy_feedback/daily_datasets/jra55_djb_ep.nc')

# define variables as same shape
ubar = ds.u.mean()
div1 = ds.div1.mean(('time'))
div2 = ds.div2.mean(('time'))

# define function
def correlation_array(da1, da2): 
    
    """
    Input: two Xarray DataArrays of same shape
    
    Output: a NumPy array of correlation coefficients,
            of same shape the input DataArrays
    """
    
    # create array of desired shape
    da_corr = np.zeros((len(da1[:,0]), len(da1[0,:])))
    
    pdb.set_trace()
    
    # loop through each variable
    # on each row, do each column entry
    for row in range(len(da1[:,0])):
        for col in range(len(da1[0,:])):
            
            # calculate correlation coefficient
            corr = xr.corr(da1[row, col], da2[row, col])
            
            pdb.set_trace()
            
            # save coefficient to respective data point
            da_corr[row, col] = corr
            
    return da_corr

dac = correlation_array(ubar, div1)