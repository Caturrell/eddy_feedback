import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


import functions.aos_functions as aos


def zonal_mean_zonal_wind(ubar, cmap='sns.coolwarm', yscale='log', levels=21, yincrease=False, figsize=(8,5), winter=True):
    
    """
    Input: Xarray dataArray containing ubar
    
    Output: Countour plot showing zonal-mean zonal wind
    """
    
    # import custom colour map
    if cmap == 'sns.coolwarm':
        import seaborn as sns
        coolwarm = sns.color_palette("coolwarm", as_cmap=True)
        cmap = coolwarm
    
    # plot it
    plt.figure(figsize=figsize)
    
    if winter == True:
        plt.contourf(ubar.lat.values, ubar.level.values, ubar,
                 cmap=cmap, levels=levels, extend='both')
        plt.colorbar(location='bottom', orientation='horizontal', shrink=0.5,
             label='Wind speed (m/s)', ticks=[-45, -30, -15, 0, 15, 30, 45], extend='both')
    else:
        plt.contourf(ubar.lat.values, ubar.level.values, ubar,
                 cmap=cmap, levels=levels, extend='both', vmin=-45, vmax=45)
        plt.colorbar(location='bottom', orientation='horizontal', shrink=0.5,
             label='Wind speed (m/s)')
    
    plt.yscale(yscale)
    
    if yincrease == False:
        plt.gca().invert_yaxis()
    
    plt.xlabel('Latitude ($^\\circ$N)')
    
    if yscale == 'log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
    
    plt.title('Zonal-mean zonal wind')
    plt.show()


# Reproduce Nakamura plot with EP flux arrows and zonal-mean zonal wind
def nakamura_plot_DJF(ds, label, levels=21, do_ubar=True, skip_lat=8, skip_pres=2, yscale='log'):
    
    """
    Input: Xarray DataSet containing u,v,t for DJF
            - dims: (time, level, lat, lon)
    
    Output: Plot showing zonal-mean zonal wind
            and EP flux arrows
    """
    
    # check for EP fluxes in dataset
    if not isinstance(ds['ep1'], xr.DataArray):
        ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ds.u, ds.v, ds.t, 
                                                        lon='lon', lat='lat', pres='level', time='time', 
                                                        do_ubar=do_ubar)
        
        ds['ep1'] = (ep1.dims, ep1.values)
        ds['ep2'] = (ep2.dims, ep2.values)
        ds['div1'] = (div1.dims, div1.values)
        ds['div2'] = (div2.dims, div2.values)
    
    # check for ubar in dataset
    if not isinstance(ds['ubar'], xr.DataArray):
        ds['ubar'] = ds.u.mean(('lon', 'time'))
        
    ## PLOTTING TIME
    
    
    # skip variables
    skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )

    #    set variables
    lat = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
    p = ds.level.isel(dict(level=slice(None, None, skip_pres)))
    Fphi = ds.ep1.mean(('time')).isel(skip)
    Fp = ds.ep2.mean(('time')).isel(skip)
    
    ubar = ds.ubar

    fig, ax = plt.subplots(figsize=(9,5))

    import seaborn as sns
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)

    plt.contourf(ds.lat.values, ds.level.values, ubar,
              cmap=coolwarm, levels=levels)
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.5,
             label='Wind speed (m/s)')

    aos.PlotEPfluxArrows(lat, p, Fphi, Fp,
                     fig, ax, pivot='mid', yscale=yscale)
    plt.title(f'{label}')
    plt.xlabel('Latitude ($^\\circ$N)')
    
    if yscale=='log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
        
    plt.show()
    


#-----------------------------------------------------------------------------------

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
