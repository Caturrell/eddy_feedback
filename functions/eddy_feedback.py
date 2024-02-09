import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import sys
sys.path.append('/home/users/cturrell/documents/eddy_feedback')
import functions.aos_functions as aos

#======================================================================================================================================

#----------------------
# DATASET MANIPULATION
#---------------------- 

# Rename dimensions in Dataset and check if ds contains Isca variable notation
def find_rename_variables(ds):
    
    """
    Input: Xarray Dataset with variety of dimension labels
            - searches for 4 in particular. Look at aos function
               for more details
    
    Output: Xarray Dataset with required variable name changes
            - Checks for Isca labelling
    """
    
    if 'ucomp' or 'ua' in ds:
        ds = rename_variables(ds)
    
    # search for dimension labels
    dims = aos.FindCoordNames(ds)
    
    # rename variables usinf dict
    rename = {dims['lon']: 'lon', dims['lat']: 'lat', dims['pres']: 'level'}
    
    # rename dataset
    ds = ds.rename(rename)
    
    return ds


# Rename dimensions in Isca to suit my function needs
def rename_variables(ds):
    
    """
    Input: Xarray Dataset produced by Isca simulation
            - dimensions: (time, lon, lat, pfull)
            - variables: ucomp, vcomp, temp
    
    Output: Xarray DataSet with required name changes
            - dimensions: (time, lon, lat, level)
            - variables: u,v,t
    """
    
    # if-statement for Isca data
    if 'ucomp' in ds:
        # Set renaming dict
        rename = {'ucomp': 'u', 'vcomp': 'v', 'temp': 't'}
    
    # if-statement for PAMIP data    
    if 'ua' in ds:
        # Set renaming dict
        rename = {'ua': 'u', 'va': 'v', 'ta': 't'}
    
    # apply changes
    ds = ds.rename(rename)
    
    return ds

# Subset to seasonal datasets
def seasonal_dataset(ds, season='djf', save_ds=False, save_location='./ds.nc'):
    
    """ 
    Input: Xarray dataset for full year
    
    Output: Xarray dataset with required season data 
    
    """
    
    import datetime as dt
    
    # DJF (winter) sub-dataset 
    if season == 'djf': 
        ds = ds.sel(time=ds.time.dt.month.isin([12, 1, 2]))
        
    # MAM (spring) dataset
    if season == 'mam':
        ds = ds.sel(time=ds.time.dt.month.isin([3, 4, 5])) 
    
    # JJA (summer) dataset    
    if season == 'jja':
        ds = ds.sel(time=ds.time.dt.month.isin([6, 7, 8]))
        
    # SON (autumn) dataset    
    if season == 'son':
        ds = ds.sel(time=ds.time.dt.month.isin([9, 10 ,11]))
    
    # save new dataset if wanted    
    if save_ds == True:
        ds.to_netcdf(f'{save_location}')    
        
    return ds 

# sort data into interannual mean for reanalysis data
def interannual(da):
    
    # average longitude if not already done
    if 'lon' in da.dims:
        da = da.mean('lon')
        
    da = da.groupby('time.year').mean('time').load()
    
    return da 
    
    

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
    
    if 'lat' and 'lon' and 'level' and 'u' and 'v' and 't' not in ds:
        ds = find_rename_variables(ds)
    
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
    
    ## CONDITIONS

    # ensure variables are named correctly
    if 'lat' and 'lon' and 'level' and 'u' and 'v' and 't' not in ds:
        ds = find_rename_variables(ds)   
    
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
def plot_ubar(ds, label='Zonal-mean zonal wind', figsize=(9,5), latitude=None, top_atmos=0., show_rect=False,
              orientation='horizontal', location='bottom', extend='both', shrink=0.5,
              levels=21, yincrease=False, yscale='linear', round_sf=None, savefig=False, fig_label=None):
              
    
    """
    Input: Xarray dataset
            - dim labels: (time, lon, lat, level)
    
    Output: Countour plot showing zonal-mean zonal wind
    """
    
    ## CONDITIONS
    
    # ensure variables are named correctly
    if 'lat' and 'lon' and 'level' and 'u' and 'v' and 't' not in ds:
        ds = find_rename_variables(ds)  
    
    # Check to see if ubar is in DataSet
    if not 'ubar' in ds:
        ds = calculate_ubar(ds)
        
    # default is both hemispheres
    if latitude == 'NH':
        ds = ds.where( ds.lat >= 0., drop=True )
        figsize=(4,5)
        orientation='vertical'
        location='right'
        shrink=0.8
    elif latitude == 'SH':
        ds = ds.where( ds.lat <= 0., drop=True ) 
        figsize=(4,5)
        orientation='vertical'
        location='right'
        shrink=0.8
    
    # exclude stratosphere-ish
    ds = ds.where( ds.level >= top_atmos, drop=True )
    
    
    #-------------------------------------------------------------------
    
    ## PRE-PLOTTING STUFF
    
    # define ubar dataArray
    ubar = ds.ubar
    
    # calculate mean absolute value of max and min
    max_value = np.nanmax(ds.ubar.values)
    value = round(max_value, round_sf)
        
    
    # set linspace levels
    lvl = np.linspace(-value, value, levels)
    ticks = [-value, -value/2, 0, value/2, value]
    
    
    # set cmap for plotting
    import seaborn as sns
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)

    
    #-------------------------------------------------------------------
    
    ## PLOTTING TIME
    
    # set figure
    plt.figure(figsize=figsize)
    
    plt.contourf(ubar.lat.values, ubar.level.values, ubar,
                 cmap=coolwarm, levels=lvl, extend=extend)
    plt.colorbar(location=location, orientation=orientation, shrink=shrink,
             label='Wind speed (m/s)', extend=extend, ticks=ticks) 
    
    plt.title(f'{label}')
    plt.xlabel('Latitude ($^\\circ$N)')
    plt.yscale(yscale)
    
    # set direction of y-axis
    if yincrease == False:
        plt.gca().invert_yaxis()
    
    # set log or linear scale
    if yscale == 'log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
    
    # save figure, if required
    if savefig == True:
        plt.savefig(f'./plots/{fig_label}.png')
        
    # show EFP box
    if show_rect == True:
        import matplotlib.patches as patches
        
        rect = patches.Rectangle((25., 600.), 50, -400, 
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.show()
    
    
#--------------------------------------------------------------------------------------------------------------------------------


# Plot zonal-mean zonal wind with EP flux arrows
def plot_ubar_epflux(ds, label='Meridional plane zonal wind and EP flux', figsize=(9,5), latitude=None, top_atmos=0.,
                     orientation='horizontal', location='bottom', extend='both', shrink=0.5, levels=21,
                     skip_lat=1, skip_pres=1, yscale='linear', primitive=True,
                     round_sf=None, savefig=False, fig_label=None):
    
    """
    Input: Xarray DataSet containing u,v,t for DJF
            - dims: (time, level, lat, lon)
    
    Output: Plot showing zonal-mean zonal wind
            and EP flux arrows
    """
    
    ## CONDITIONS
    
    # ensure variables are named correctly
    if 'lat' and 'lon' and 'level' and 'u' and 'v' and 't' not in ds:
        ds = find_rename_variables(ds)  
    
    # Check to see if EP fluxes are in DataSet
    if not 'ep1' in ds:
        ds = calculate_epfluxes_ubar(ds, primitive=primitive)
        
    ## default is both hemispheres
    if latitude == 'NH':
        ds = ds.where( ds.lat >= 0., drop=True )
        orientation='vertical'
        location='right'
        shrink=0.8
    elif latitude == 'SH':
        ds = ds.where( ds.lat <= 0., drop=True ) 
        orientation='vertical'
        location='right' 
        shrink=0.8
    
    # exclude stratosphere-ish
    ds = ds.where( ds.level >= top_atmos, drop=True )
    
    #-------------------------------------------------------------------
        
    ## PRE-PLOTTING STUFF
    
    # define ubar
    ubar = ds.ubar
    
    # calculate mean absolute value of max and min
    max_value = np.nanmax(ds.ubar.values)
    value = round(max_value, round_sf)
        
    
    # set linspace levels
    lvl = np.linspace(-value, value, levels)
    ticks = [-value, -value/2, 0, value/2, value]
    
    
    # skip variables
    skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )

    #    set variables
    lat = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
    p = ds.level.isel(dict(level=slice(None, None, skip_pres)))
    Fphi = ds.ep1.mean(('time')).isel(skip)
    Fp = ds.ep2.mean(('time')).isel(skip)
    
    
    #-------------------------------------------------------------------
    
    ## PLOTTING TIME

    # Set figure
    fig, ax = plt.subplots(figsize=figsize)

    # set cmap from seaborn
    import seaborn as sns
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)

    plt.contourf(ds.lat.values, ds.level.values, ubar,
              cmap=coolwarm, levels=lvl, extend=extend)
    plt.colorbar(location=location, orientation=orientation, shrink=shrink,
             label='Wind speed (m/s)', extend=extend, ticks=ticks)

    aos.PlotEPfluxArrows(lat, p, Fphi, Fp,
                     fig, ax, pivot='mid', yscale=yscale)
    plt.title(f'{label}')
    plt.xlabel('Latitude ($^\\circ$N)')
    
    # set whether log or linear scale
    if yscale=='log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
    
    # save figure if required
    if savefig == True:
        plt.savefig(f'./plots/{fig_label}.png')
        
    plt.show()
    
    
#--------------------------------------------------------------------------------------------------------------------------------


    
# plot EP fluxes and northward divergence
def plot_epfluxes_div(ds, label='EP flux and northward divergence of EP Flux', figsize=(9,5), yscale='linear', primitive=True, 
                      latitude=None, lat_slice=slice(None,None), top_atmos=100., skip_lat=1, skip_pres=1,
                      orientation='horizontal', location='bottom', extend='both', shrink=0.5, levels=21,
                      plot_arrows=True, show_rect=False, round_sf=None, savefig=False, fig_label=None):
    
    """
    Input: Xarray Dataset containing u,v,t
    
    Output: Plot of EP flux vector arrows and
            divergence as contour plot
    """
    
    ## CONDITIONS
    
    # ensure variables are named correctly
    if 'lat' and 'lon' and 'level' and 'u' and 'v' and 't' not in ds:
        ds = find_rename_variables(ds)  
    
    # Check to see if EP fluxes are in DataSet
    if not 'ep1' in ds:
        ds = calculate_epfluxes_ubar(ds, primitive=primitive)
        
    # reanalysis has unrealistic values at poles, so cut off ends
    ds = ds.isel(lat=lat_slice)
    
    
    ## default is both hemispheres
    if latitude == 'NH':
        ds = ds.where( ds.lat >= 0., drop=True )
        figsize=(4,5)
        orientation='vertical'
        location='right'
        shrink=0.8
    elif latitude == 'SH':
        ds = ds.where( ds.lat <= 0., drop=True ) 
        figsize=(4,5)
        orientation='vertical'
        location='right' 
        shrink=0.8
    
    # exclude stratosphere-ish
    ds = ds.where( ds.level >= top_atmos, drop=True ) 
        
    
    #-------------------------------------------------------------------
    
    ## PRE-PLOTTING STUFF
    
    # Set divergence of div1 and remove outliers
    div1 = ds.div1.mean(('time'))
    div1 = div1.where(abs(div1) < 1e2)
    
    # calculate mean absolute value of max and min
    max_value = np.nanmax(div1.values)
    value = round(max_value, round_sf)
        
    
    # set linspace levels
    lvl = np.linspace(-value, value, levels)
    ticks = [-value, -value/2, 0, value/2, value]
    
    
    # skip variables
    skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )

    #    set variables
    lat = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
    p = ds.level.isel(dict(level=slice(None, None, skip_pres)))
    Fphi = ds.ep1.mean(('time')).isel(skip)
    Fp = ds.ep2.mean(('time')).isel(skip)
    
    import seaborn as sns
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    
    
    #-------------------------------------------------------------------
    
    ## PLOTTING TIME
    
    # set figures
    fig, ax = plt.subplots(figsize=figsize)

    plt.contourf(ds.lat.values, ds.level.values, div1,
              cmap=coolwarm, levels=lvl, extend=extend)
    plt.colorbar(location=location, orientation=orientation, shrink=shrink,
             label='Wind speed (m/s)', extend=extend, ticks=ticks)

    if plot_arrows == True:
        aos.PlotEPfluxArrows(lat, p, Fphi, Fp,
                     fig, ax, pivot='mid', yscale=yscale)
    else:
        plt.gca().invert_yaxis()
        plt.yscale(yscale)
    
    plt.title(f'{label}')
    plt.xlabel('Latitude ($^\\circ$N)')
    
    # set whether log or linear scale
    if yscale=='log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
    
    # save figure if required
    if savefig == True:
        plt.savefig(f'./plots/{fig_label}.png')
        
    # show EFP box
    if show_rect == True:
        import matplotlib.patches as patches
        
        rect = patches.Rectangle((25., 600.), 50, -400, 
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)
        
    plt.show()



#======================================================================================================================================

#-------------------- 
# STATISTICAL PLOTS
#--------------------


def correlation_contourf(ds, label='DJF', top_atmos=10., reanalysis=True, hemisphere='NH',
                         show_div2=False, logscale=True, show_rect=True, primitive=True):
    
    """"
    Input: dataset that contains ep fluxes data
            - with variables: (time, level, lat, lon)
    
    Output: contourf plot matching Fig.6 in Smith et al., 2022 
    """
    
    ## CONDITIONS
    
    # ensure variables are named correctly
    if 'lat' and 'lon' and 'level' and 'u' and 'v' and 't' not in ds:
        ds = find_rename_variables(ds)  
    
    # Check to see if EP fluxes are in DataSet
    if not 'ep1' in ds:
        ds = calculate_epfluxes_ubar(ds, primitive=primitive) 
        
    # choose hemisphere
    if hemisphere == 'SH':
        # set southern hermisphere
        ds = ds.where( ds.lat <= 0, drop=True )
    else:
        # set northern hemisphere
        ds = ds.where( ds.lat >= 0, drop=True )
    
    # cut off stratosphere
    ds = ds.where( ds.level >= top_atmos, drop=True )
    
    #------------------------------------------------------------------
    
    ## SET UP TIME
    
    # remove unwanted variables
    vars = ['u', 'div1', 'div2']
    ds = ds[vars]
    
    # set variables and save them
    ubar = ds.u.mean(('lon'))
    div1 = ds.div1
    div2 = ds.div2
    
    if reanalysis == True:
        # separate time into annual means
        # and use .load() to force the calculation now
        ubar = ubar.groupby('time.year').mean('time').load()
        div1 = div1.groupby('time.year').mean('time').load()
        div2 = div2.groupby('time.year').mean('time').load()
    else:
        # separate time into annual means
        ubar = ubar.load()
        div1 = div1.load()
        div2 = div2.load()
    
    # choose which variable; default: div1
    if show_div2==True:
        if reanalysis == True:
            corr = xr.corr(ubar, div2, dim='year') 
        else:
            corr = xr.corr(ubar, div2, dim='time')
        title_name = '\\nabla_p F_p'
        figgy = (6,7)
    else:
        if reanalysis == True:
            corr = xr.corr(ubar, div1, dim='year')
        else:
            corr = xr.corr(ubar, div1, dim='time')
        title_name = '\\nabla_{\\phi} F_{\\phi}'
        figgy = (6,6)
        
    import matplotlib.patches as patches

    plt.figure(figsize=figgy)

    plt.contourf(ds.lat.values, ds.level.values, corr, cmap='RdBu_r', levels=np.linspace(-0.9,0.9,19),
             extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.75, label='correlation',
             extend='both', ticks=[-0.6,-0.2,0.2,0.6])
    plt.gca().invert_yaxis()
    
    if logscale==True:
        plt.yscale('log')
        

    plt.xlabel('Latitude $(^\\circ N)$')
    plt.ylabel('Log pressure (hPa)')
    plt.title('$Corr(\\bar{{u}}, {0})$ - {1}'.format(title_name, label))

    if show_rect == True:
        rect = patches.Rectangle((25., 600.), 50, -400, 
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()
    

#--------------------------------------------------------------------------------------------------------------------------------
    
def plot_variance(ds, variable='ubar', top_atmos=100.,
                  figsize=(9,5), orientation='horizontal', location='bottom',
                  latitude=None, primitive=True, reanalysis=True,
                  logscale=True, show_rect=True):
    
    """
    Input: DataArray of required variable with three dimensions
            - Usually (time, level, lat)
            
    Output: Contour plot showing (lat, level) variance
    
    """
    
    ## CONDITIONS
    
    # ensure variables are named correctly
    if 'lat' and 'lon' and 'level' and 'u' and 'v' and 't' not in ds:
        ds = find_rename_variables(ds)  
    
    # Check to see if EP fluxes are in DataSet
    if not 'ep1' in ds:
        ds = calculate_epfluxes_ubar(ds, primitive=primitive)

    # remove poles for div1 variable because massive
    if variable == 'div1':
        ds = ds.isel(lat=slice(1,72)) 
    
    # set top of atmosphere
    ds = ds.where( ds.level >= top_atmos, drop=True )
    
    # Choose hemisphere, if required
    if latitude == 'NH':
        ds = ds.where( ds.lat >= 0., drop=True )
        figsize=(4,5)
        orientation='vertical'
        location='right'
    if latitude == 'SH':
        ds = ds.where( ds.lat <= 0., drop=True )
        figsize=(4,5)
        orientation='vertical'
        location='right'

        
    #-------------------------------------------------------------------
        
    ## PRE-PLOTTING SET UP
        
    # Choose required DataArray
    if variable == 'ubar':
        da = ds['u'].mean(('lon'))
    else:
        da = ds[variable]

    # take interannual time average if reanalysis set
    # and specify which variable to find the variance over
    if reanalysis == True:
        da = interannual(da)
        var = da.var(dim='year').load()
    else:
        var = da.var(dim='time').load()

    # find max value
    max_value = np.max(var).round(1)

    #-------------------------------------------------------------------
        
    ## PLOTTING TIME
        
    # set up figure
    plt.figure(figsize=figsize)

    plt.contourf(var.lat.values, var.level.values, var, cmap='Greens',
                 levels=np.linspace(0, max_value, 21), extend='both')
    plt.colorbar(location=location, orientation=orientation, shrink=0.75, 
                 label='variance', extend='both')
    
    plt.gca().invert_yaxis()
    # choose log or linear scale
    if logscale==True:
        plt.yscale('log')
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.yscale('linear')
        plt.ylabel('Pressure (hPa)')

    plt.xlabel('Latitude $(^\\circ N)$')
    plt.title(f'Variance of {variable}')

    if show_rect == True:
        import matplotlib.patches as patches 
        rect = patches.Rectangle((25., 600.), 50, -400, 
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()
    


if __name__ == '__main__':
    
    # era5
    ds = xr.open_mfdataset('/gws/nopw/j04/arctic_connect/cturrell/era5_data/era5daily_djf_uvt.nc', 
                            parallel=True, chunks={'time': 31})
    
    ds = calculate_epfluxes_ubar(ds)
    
    # save dataset
    ds.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/era5_data/era5daily_djf_uvt_ep.nc')