import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


import functions.aos_functions as aos

#======================================================================================================================================

#------------------
# DATASET RENAMING
#------------------

# Rename dimensions in Isca to suit my function needs
def rename_isca_variables(ds):
    
    """
    Input: Xarray DataSet produced by Isca simulation
            - dimensions: (time, lon, lat, pfull)
            - variables: ucomp, vcomp, temp
    
    Output: Xarray DataSet with required name changes
            - dimensions: (time, lon, lat, level)
            - variables: u,v,t
    """
    
    # Set renaming dict
    rename = {'pfull': 'level', 'ucomp': 'u', 'vcomp': 'v', 'temp': 't'}
    
    ds = ds.rename(rename)
    
    return ds



#======================================================================================================================================

#----------------------
# DATASET CALCULATIONS
#----------------------


# Calculate zonal-mean zonal wind
def calculate_ubar(ds):
    
    """
    Input: Xarray dataset
            - dim labels: (time, lon, lat, level)
            - variable: u
            
    Output: Xarray dataset with zonal-mean zonal wind 
            calculated and added as variable
    """
    
    if 'pfull' in ds:
        ds = rename_isca_variables(ds)
    
    # Calculate 
    ds['ubar'] = ds.u.mean(('time', 'lon'))
    
    return ds

# Calculate EP fluxes
def calculate_epfluxes_ubar(ds, primitive=True):
    
    """
    Input: Xarray dataset
            - dim labels: (time, lon, lat, level)
            - variable: u,v,t
            
    Output: Xarray dataset with EP fluxes calculated
            and optional calculate ubar
    """

    # ensure variables are named correctly
    if 'pfull' in ds:
        ds = rename_isca_variables(ds)    
    
    # check if ubar is in dataset also
    if not 'ubar' in ds:
        ds = calculate_ubar(ds)
        
    
    import functions.aos_functions as aos
    
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ds.u, ds.v, ds.t, do_ubar=primitive)
    
    ds['ep1'] = (ep1.dims, ep1.values)
    ds['ep2'] = (ep2.dims, ep2.values)
    ds['div1'] = (div1.dims, div1.values)
    ds['div2'] = (div2.dims, div2.values)
    
    return ds


#======================================================================================================================================

#---------------------------
# PLOTTING MERIDIONAL PLANE
#---------------------------

# Plot zonal-mean zonal wind on meridional plane
def plot_ubar(ds, label='Zonal-mean zonal wind', cmap='sns.coolwarm', 
              levels=21, yincrease=False, yscale='linear', figsize=(8,5)):
    
    """
    Input: Xarray dataset
            - dim labels: (time, lon, lat, level)
    
    Output: Countour plot showing zonal-mean zonal wind
    """
    
    # Check to see if EP fluxes are in DataSet
    if not 'ubar' in ds:
        ds = calculate_ubar(ds)
    
    # define ubar dataArray
    ubar = ds.ubar
    
    # import custom colour map
    if cmap == 'sns.coolwarm':
        import seaborn as sns
        coolwarm = sns.color_palette("coolwarm", as_cmap=True)
        cmap = coolwarm
        
    # calculate max value
    
    # plot it
    plt.figure(figsize=figsize)
    
    plt.contourf(ubar.lat.values, ubar.level.values, ubar,
                 cmap=cmap, levels=levels, extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.5,
             label='Wind speed (m/s)', extend='both')
    
    plt.yscale(yscale)
    
    if yincrease == False:
        plt.gca().invert_yaxis()
    
    plt.xlabel('Latitude ($^\\circ$N)')
    
    if yscale == 'log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
    
    plt.title(f'{label}')
    plt.show()
    
    
#--------------------------------------------------------------------------------------------------------------------------------


# Plot zonal-mean zonal wind with EP flux arrows
def plot_ubar_epflux(ds, label='Meridional plane zonal wind and EP flux', 
                     levels=21, skip_lat=1, skip_pres=1, yscale='linear', primitive=True):
    
    """
    Input: Xarray DataSet containing u,v,t for DJF
            - dims: (time, level, lat, lon)
    
    Output: Plot showing zonal-mean zonal wind
            and EP flux arrows
    """
    
    # Check to see if EP fluxes are in DataSet
    if not 'ep1' in ds:
        ds = calculate_epfluxes_ubar(ds, primitive=primitive)
        
        
    ## PLOTTING TIME
    
    # skip variables
    skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )

    #    set variables
    lat = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
    p = ds.level.isel(dict(level=slice(None, None, skip_pres)))
    Fphi = ds.ep1.mean(('time')).isel(skip)
    Fp = ds.ep2.mean(('time')).isel(skip)
    
    # define ubar
    ubar = ds.ubar


    # Set figure
    fig, ax = plt.subplots(figsize=(9,5))

    import seaborn as sns
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)

    plt.contourf(ds.lat.values, ds.level.values, ubar,
              cmap=coolwarm, levels=levels, extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.5,
             label='Wind speed (m/s)', extend='both')

    aos.PlotEPfluxArrows(lat, p, Fphi, Fp,
                     fig, ax, pivot='mid', yscale=yscale)
    plt.title(f'{label}')
    plt.xlabel('Latitude ($^\\circ$N)')
    
    if yscale=='log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
        
    plt.show()
    
    
#--------------------------------------------------------------------------------------------------------------------------------


    
# plot EP fluxes and northward divergence
def plot_epfluxes_div(ds, label='EP flux and northward divergence of EP Flux', latitude='both', top_atmos=100., 
                      levels=21, skip_lat=1, skip_pres=1, yscale='linear', primitive=True):
    
    # Check to see if EP fluxes are in DataSet
    if not 'ep1' in ds:
        ds = calculate_epfluxes_ubar(ds, primitive=primitive)
    
    
    # default is both hemispheres
    if latitude == 'NH':
        ds = ds.where( ds.lat >= 0., drop=True )
    elif latitude == 'SH':
        ds = ds.where( ds.lat <= 0., drop=True ) 
    
    # exclude stratosphere-ish
    ds = ds.where( ds.level >= top_atmos, drop=True ) 
    
    
    # Set divergence of div1 and remove outliers
    div1 = ds.div1.mean(('time'))
    div1 = div1.where(abs(div1) < 1e2) 
    
    # skip variables
    skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )

    #    set variables
    lat = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
    p = ds.level.isel(dict(level=slice(None, None, skip_pres)))
    Fphi = ds.ep1.mean(('time')).isel(skip)
    Fp = ds.ep2.mean(('time')).isel(skip)
    
    fig, ax = plt.subplots(figsize=(9,5))

    import seaborn as sns
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)

    plt.contourf(ds.lat.values, ds.level.values, div1,
              cmap=coolwarm, levels=levels, extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.5,
             label='Wind speed (m/s)', extend='both')

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


def correlation_contourf(ds, show_div2=False, logscale=True, show_rect=True):
    
    """"
    Input: dataset that contains ep fluxes data
            - with variables: (time, level, lat, lon)
    
    Output: contourf plot matching Fig.6 in Smith et al., 2022
    """
        
    ds = ds.isel(dict( level=slice(10,37) ))
        
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

    if show_rect == True:
        rect = patches.Rectangle((25., 600.), 50, -400, 
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()
