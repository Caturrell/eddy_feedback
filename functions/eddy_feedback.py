import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# correlation on a grid function
def correlation_array(da1, da2, show_progress=False):
    
    """
    Input: two Xarray DataArrays of same shape (time,level,lat)
    
    Output: a NumPy array of correlation coefficients,
            of shape (level, lat)
            
            
    !!! Might need to check .load() otherwise will run
        for VERY long time
    
    """
    
    # show progress bar
    if show_progress==True:
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
    else:
        # create array of desired shape
        da_corr = np.zeros((len(da1[0,:,0]), len(da1[0,0,:])))
    
        # loop through each variable
        # on each row, do each column entry
        for i in range(len(da1[0,:,0])):
            for j in range(len(da1[0,0,:])):
            
                # calculate correlation coefficient
                corr = np.corrcoef(da1[:,i, j], da2[:,i, j])  
                # save coefficient to respective data point
                da_corr[i, j] = corr[0,1]
        
            
    return da_corr 


def correlation_contourf(ds, show_div2=False, logscale=True):
    
    """"
    Input: dataset that contains ep fluxes data
            - with variables: (time, level, lat, lon)
    
    Output: contourf plot matching Fig.6 in Smith et al., 2022
    """
    
    # check if ubar is a variable, if not then calculate
    if not isinstance(ds['ubar'], xr.DataArray):
        ds['ubar'] = ds.u.mean(('lon'))
        
    # set variables and save them
    ubar = ds.u.mean(('lon'))
    div1 = ds.div1
    div2 = ds.div2
    
    # separate time into annual means
    ubar = ubar.groupby('time.year').mean('time').load()
    div1 = div1.groupby('time.year').mean('time').load()
    div2 = div2.groupby('time.year').mean('time').load()
    
    # choose which variable; default: div1
    if show_div2==True:
        corr = correlation_array(ubar, div2)
        title_name = '\\nabla_p F_p'
        figgy = (6,7)
    else:
        corr = correlation_array(ubar, div1)
        title_name = '\\nabla_{\\phi} F_{\\phi}'
        figgy = (6,6)
        
    import matplotlib.patches as patches

    plt.figure(figsize=figgy)

    plt.contourf(ds.lat.values, ds.level.values, corr, cmap='RdBu_r', levels=15,
             extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.75, label='correlation',
             extend='both', ticks=[-0.6,-0.2,0.2,0.6])
    plt.gca().invert_yaxis()
    
    if logscale==True:
        plt.yscale('log')

    plt.xlabel('Latitude $(^\\circ N)$')
    plt.ylabel('Log pressure (hPa)')
    plt.title('(a) $Corr(\\bar{{u}}, {0})$'.format(title_name))

    rect = patches.Rectangle((25., 600.), 50, -400, 
                         fill=False, linewidth=2)
    plt.gca().add_patch(rect)

    plt.show()
