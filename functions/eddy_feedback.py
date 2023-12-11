import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# correlation on a grid function
def correlation_array(da1, da2):
    
    """
    Input: two Xarray DataArrays of same shape (time,level,lat)
    
    Output: a NumPy array of correlation coefficients,
            of shape (level, lat)
            
            
    !!! Might need to check .load() otherwise will run
        for VERY long time
    
    """
    
    # show progress bar
    from tqdm import tqdm
    
    # create array of desired shape
    da_corr = np.zeros((len(da1[0,:,0]), len(da1[0,0,:])))
    
    # loop through each variable
    # on each row, do each column entry
    for i in tqdm(range(len(da1[0,:,0]))):
        for j in range(len(da1[0,0,:])):
            
            # calculate correlation coefficient
            corr = np.corrcoef(da1[:,i, j], da2[:,i, j])
            
            # save coefficient to respective data point
            da_corr[i, j] = corr[0,1]
            
    return da_corr 