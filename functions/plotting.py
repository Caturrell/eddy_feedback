""" 
    Various functions used for plotting eddy feedback-related figures.
"""
# pylint: disable=wrong-import-position
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=invalid-name

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import xarray as xr

## JASMIN SERVERS
# sys.path.append('/home/users/cturrell/documents/eddy_feedback')

## MATHS SERVERS
sys.path.append('/home/links/ct715/eddy_feedback/')


import functions.aos_functions as aos
import functions.data_wrangling as data
import functions.eddy_feedback as ef


#==================================================================================================

#---------------------------
# PLOTTING MERIDIONAL PLANE
#---------------------------

# Plot zonal-mean zonal wind on meridional plane
def plot_ubar(ds, label='Zonal-mean zonal wind', figsize=(9,5),
              latitude=None, top_atmos=0., show_rect=False, orientation='horizontal',
              location='bottom', extend='both', shrink=0.5, levels=21, yincrease=False,
              yscale='linear', round_sf=None, savefig=False, fig_label=None):


    """
    Input: Xarray dataset
            - dim labels: (time, lon, lat, level)
    
    Output: Countour plot showing zonal-mean zonal wind
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'lon'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    # Check to see if ubar is in DataSet and define it
    if not 'ubar' in ds:
        ds = ef.calculate_ubar(ds)
    # define ubar
    ubar = ds.ubar

    # default is both hemispheres
    if latitude == 'NH':
        ubar = ubar.where( ubar.lat >= 0., drop=True )
        figsize=(4,5)
        orientation='vertical'
        location='right'
        shrink=0.8
    elif latitude == 'SH':
        ubar = ubar.where( ubar.lat <= 0., drop=True )
        figsize=(4,5)
        orientation='vertical'
        location='right'
        shrink=0.8

    # exclude stratosphere-ish
    ubar = ubar.where( ubar.level >= top_atmos, drop=True )


    #-------------------------------------------------------------------

    ## PRE-PLOTTING STUFF

    # calculate time average
    ubar = ubar.mean('time')

    # calculate mean absolute value of max and min
    max_value = np.nanmax(ubar.values)
    value = round(max_value, round_sf)


    # set linspace levels
    lvl = np.linspace(-value, value, levels)
    ticks = [-value, -value/2, 0, value/2, value]


    #-------------------------------------------------------------------

    ## PLOTTING TIME

    # set figure
    plt.figure(figsize=figsize)

    plt.contourf(ubar.lat.values, ubar.level.values, ubar,
                 cmap='coolwarm', levels=lvl, extend=extend)
    plt.colorbar(location=location, orientation=orientation, shrink=shrink,
             label='Wind speed (m/s)', extend=extend, ticks=ticks)

    plt.title(f'{label}')
    plt.xlabel('Latitude ($^\\circ$N)')
    plt.yscale(yscale)

    # set direction of y-axis
    if yincrease is False:
        plt.gca().invert_yaxis()

    # set log or linear scale
    if yscale == 'log':
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')

    # save figure, if required
    if savefig:
        plt.savefig(f'./plots/{fig_label}.png')

    # show EFP box
    if show_rect:

        rect = patches.Rectangle((25., 600.), 50, -400,
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()


#--------------------------------------------------------------------------------------------------


# Plot zonal-mean zonal wind with EP flux arrows
def plot_ubar_epflux(ds, label='Meridional plane zonal wind and EP flux', figsize=(9,5),
                     latitude=None, top_atmos=0., orientation='horizontal', location='bottom',
                     extend='both', shrink=0.5, levels=21, skip_lat=1, skip_pres=1, yscale='linear',
                     round_sf=None, savefig=False, fig_label=None, season=None, plot_arrows=True,
                     arrow_loc=(0.85,1.05)):

    """
    Input: Xarray DataSet containing ubar and epfluxes
            - dims: (time, level, lat)
    
    Output: Plot showing zonal-mean zonal wind
            and EP flux arrows
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'lon'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    if season is not None:
        ds = data.seasonal_mean(ds, season=season)

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
    ubar = ds.u.mean(('time', 'lon'))

    # calculate mean absolute value of max and min
    max_value = np.nanmax(ubar.values)
    value = round(max_value, round_sf)


    # set linspace levels
    lvl = np.linspace(-value, value, levels)
    ticks = [-value, -value/2, 0, value/2, value]


    # skip variables
    # skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )
    skip = {'lat':slice(None, None, skip_lat), 'level':slice(None, None, skip_pres)}

    # set variables
    # lat = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
    # p = ds.level.isel(dict(level=slice(None, None, skip_pres)))
    lat = ds.lat.isel( {'lat':slice(None, None, skip_lat)} )
    p = ds.level.isel( {'level':slice(None, None, skip_pres)} )


    #-------------------------------------------------------------------

    ## PLOTTING TIME

    # Set figure
    fig, ax = plt.subplots(figsize=figsize)

    plt.contourf(ds.lat.values, ds.level.values, ubar,
              cmap='coolwarm', levels=lvl, extend=extend)
    plt.colorbar(location=location, orientation=orientation, shrink=shrink,
             label='Wind speed (m/s)', extend=extend, ticks=ticks)

    if plot_arrows:
        Fphi = ds.ep1.mean(('time')).isel(skip)
        Fp = ds.ep2.mean(('time')).isel(skip)
        aos.PlotEPfluxArrows(lat, p, Fphi, Fp, fig, ax, 
                             pivot='mid', arrow_loc=arrow_loc, yscale=yscale)
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
    if savefig:
        plt.savefig(f'./plots/{fig_label}.png')

    plt.show()


#--------------------------------------------------------------------------------------------------


# plot EP fluxes and northward divergence
def plot_epfluxes_div(ds, label='EP flux and northward divergence of EP Flux', which_div1='div1',
                      figsize=(9,5), yscale='linear', latitude=None, remove_poles=True,
                      top_atmos=100., skip_lat=1, skip_pres=1, orientation='horizontal',
                      location='bottom', extend='both', shrink=0.5, levels=21, plot_arrows=True,
                      show_rect=False, round_sf=None, savefig=False, fig_label=None):

    """
    Input: Xarray Dataset containing u,v,t
    
    Output: Plot of EP flux vector arrows and
            divergence as contour plot
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'lon'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    # reanalysis has unrealistic values at poles, so cut off ends
    if remove_poles:
        ds = ds.isel(lat=slice(1,-1))

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
    div1 = ds[f'{which_div1}'].mean(('time'))
    div1 = div1.where(abs(div1) < 1e2)

    # calculate mean absolute value of max and min
    max_value = np.nanmax(div1.values)
    value = round(max_value, round_sf)


    # set linspace levels
    lvl = np.linspace(-value, value, levels)
    ticks = [-value, -value/2, 0, value/2, value]


    # skip variables
    # skip = dict( lat=slice(None, None, skip_lat), level=slice(None, None, skip_pres) )
    skip = {'lat':slice(None, None, skip_lat), 'level':slice(None, None, skip_pres)}

    # set variables
    # lat = ds.lat.isel(dict(lat=slice(None, None, skip_lat)))
    # p = ds.level.isel(dict(level=slice(None, None, skip_pres)))
    lat = ds.lat.isel( {'lat':slice(None, None, skip_lat)} )
    p = ds.level.isel( {'level':slice(None, None, skip_pres)} )


    #-------------------------------------------------------------------

    ## PLOTTING TIME

    # set figures
    fig, ax = plt.subplots(figsize=figsize)

    plt.contourf(ds.lat.values, ds.level.values, div1,
              cmap='coolwarm', levels=lvl, extend=extend)
    plt.colorbar(location=location, orientation=orientation, shrink=shrink,
             label='Wind speed (m/s)', extend=extend, ticks=ticks)

    if plot_arrows:
        Fphi = ds.ep1.mean(('time')).isel(skip)
        Fp = ds.ep2.mean(('time')).isel(skip)
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
    if savefig:
        plt.savefig(f'./plots/{fig_label}.png')

    # show EFP box
    if show_rect:
        rect = patches.Rectangle((25., 600.), 50, -400,
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()
    



#==================================================================================================

#--------------------
# STATISTICAL PLOTS
#--------------------

def plot_variance(ds, variable='ubar', remove_poles=False, top_atmos=100.,
                  figsize=(9,5), orientation='horizontal', location='bottom', season='djf',
                  latitude=None, logscale=True, show_rect=True):

    """
    Input: DataArray of required variable related to EFP
            - Usually (time, level, lat)
            
    Output: Contour plot showing (lat, level) variance
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    #-------------------------------------------------------------------

    ## PRE-PLOTTING SET UP

    # Choose required DataArray
    da = ds[variable]

    # take seasonal time average and calculate variance
    da = data.seasonal_mean(da, season=season)
    var = da.var(dim='time').load()

    # remove poles for div1 variable because massive
    if remove_poles:
        var = var.isel(lat=slice(1,-1))

    # choose hemisphere
    if latitude == 'NH':
        var = var.where(var.lat >= 0., drop=True)
        figsize=(4,5)
        orientation='vertical'
        location='right'
    elif latitude == 'SH':
        var = var.where(var.lat <= 0., drop=True)
        figsize=(4,5)
        orientation='vertical'
        location='right'

    # choose top of atmosphere
    var = var.where(var.level >= top_atmos, drop=True)

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
    if logscale:
        plt.yscale('log')
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.yscale('linear')
        plt.ylabel('Pressure (hPa)')

    plt.xlabel('Latitude $(^\\circ N)$')
    plt.title(f'Variance of {variable}')

    if show_rect:
        rect = patches.Rectangle((25., 600.), 50, -400,
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()


#==================================================================================================

#---------------------
# EDDY FEEDBACK PLOTS
#---------------------

def plot_reanalysis_correlation(ds, label='DJF', logscale=True, show_rect=True, latitude='NH',
                                top_atmos=10., cut_poles=False, figsize=(6,6),
                                title_name = '\\nabla_{\\phi} F_{\\phi}', take_seasonal=True,
                                which_div1='div1_pr'):
    """"
    Input: DataArrays of ubar and F_\\phi
            - Dims: (time, level, lat)
                - DATASET MUST BE FULL YEAR FOR SEASONAL MEAN 
            - Subsetted to NH or SH
            - Cut off stratosphere >10. hPa
    
    Output: contourf plot matching Fig.6a in Smith et al., 2022 
    """

    ## SET UP TIME

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    # choose hemisphere
    if latitude == 'NH':
        ds = ds.where(ds.lat >= 0., drop=True)
        rect_box = (25., 600.)
        season='djf'
    elif latitude == 'SH':
        ds = ds.where(ds.lat <= 0., drop=True)
        rect_box = (-75., 600.)
        season='jas'

    # separate time into annual means
    # and use .load() to force the calculation now
    if take_seasonal:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season=season)


    # set variables
    ubar = ds.ubar
    div1 = ds[which_div1]

    # calculate correlation using built-in Xarray function
    corr = xr.corr(div1, ubar, dim='time')

    # choose top of atmosphere
    corr = corr.where(corr.level >= top_atmos, drop=True)

    if cut_poles:
        corr = corr.where(corr.lat >= -85., drop=True)
        corr = corr.where(corr.lat <= 85., drop=True)

    #------------------------------------------------------------------

    ## PLOTTING TIME

    # Initiate plot
    plt.figure(figsize=figsize)

    # actual plotting
    plt.contourf(corr.lat.values, corr.level.values, corr, cmap='RdBu_r',
                 levels=np.linspace(-0.9,0.9,19), extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.75, label='correlation',
             extend='both', ticks=[-0.6,-0.2,0.2,0.6])

    # axis alterations
    plt.gca().invert_yaxis()
    plt.xlabel('Latitude $(^\\circ N)$')
    if logscale:
        plt.yscale('log')
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
    # plt.title('$Corr(\\bar{{u}}, {0})$ - {1}'.format(title_name, label))
    plt.title(f'$Corr(\\bar{{u}}, {title_name})$ - {label}')

    # Plot EFP box
    if show_rect:
        rect = patches.Rectangle(rect_box, 50, -400,
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()


def plot_pamip_correlation(ds, take_seasonal=True, season='djf', logscale=True, show_rect=True,
                           top_atmos=10., cut_poles=90, label='DJF'):

    """"
    Input: DataArrays of ubar and F_\\phi with PAMIP data
            - Dims: (ens_ax, time, level, lat)
    
    Output: contourf plot replicating Fig.6a in Smith et al., 2022 
    """

    ## SET UP TIME

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'ens_ax'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    if take_seasonal:
        ds = data.seasonal_dataset(ds, season=season)
        ds = ds.mean('time')

    # Subset to NH and cut top of atmos. and poles
    # Can choose latitude cut-off
    ds = ds.sel(lat=slice(0,cut_poles))

    # Cut off top of atmosphere
    ds = ds.where(ds.level > top_atmos, drop=True)

    # Calculate correlation
    corr = xr.corr(ds.div1, ds.ubar, dim='ens_ax')

    #------------------------------------------------------------------

    ## PLOTTING TIME

    # Initiate plot
    plt.figure(figsize=(6,6))

    # actual plotting
    plt.contourf(corr.lat.values, corr.level.values, corr, cmap='RdBu_r',
                 levels=np.linspace(-0.9,0.9,19), extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.75, label='correlation',
             extend='both', ticks=[-0.6,-0.2,0.2,0.6])

    # axis alterations
    plt.gca().invert_yaxis()
    plt.xlabel('Latitude $(^\\circ N)$')
    if logscale:
        plt.yscale('log')
        plt.ylabel('Log pressure (hPa)')
    else:
        plt.ylabel('Pressure (hPa)')
    plt.title(f'$Corr(\\bar{{u}}, \\nabla_{{\\phi}} F_{{\\phi}})$ - {label}')

    # Plot EFP box
    if show_rect:
        rect = patches.Rectangle((25.,600.), 50, -400,
                         fill=False, linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()
    
    
def plot_isca_correlation(ds, both_hemis=False):
    
    """"
    Input: DataArrays of ubar and F_\\phi with Isca data
            - Dims: (time, level, lat)
    
    Output: contourf plot replicating Fig.6a in Smith et al., 2022 
    """
    
    #-----------------------------------------------------------------------------------------------

    # define variables
    ubar = ds.ubar
    div1 = ds.div1

    # Calculate Pearson's correlation
    corr = xr.corr(div1, ubar, dim='time')

    if not both_hemis:
        corr = corr.sel(lat=slice(0,90))
    corr = corr.sel(level=slice(1000,10))

    # Initiate plot
    plt.figure(figsize=(6,6))

    # actual plotting
    plt.contourf(corr.lat.values, corr.level.values, corr, cmap='RdBu_r',
                    levels=np.linspace(-0.9,0.9,19), extend='both')
    plt.colorbar(location='bottom', orientation='horizontal', shrink=0.75, label='correlation',
                extend='both', ticks=[-0.6,-0.2,0.2,0.6])

    # axis alterations
    plt.gca().invert_yaxis()
    plt.xlabel('Latitude $(^\\circ N)$')
    plt.yscale('log')
    plt.ylabel('Log pressure (hPa)')
    plt.title(f'$Corr(\\bar{{u}}, \\nabla_{{\\phi}} F_{{\\phi}})$')
    from matplotlib import patches
    rect = patches.Rectangle((25.,600.), 50, -400,
                        fill=False, linewidth=2)
    plt.gca().add_patch(rect)

    plt.show()
    
def plot_correlation_components(ds, season='djf', reanalysis=False, pamip=False):
    
    ## SET UP TIME

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    if reanalysis:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season=season)
    elif pamip:
        ds = data.seasonal_dataset(ds, season=season)
    else:
        print('No data type selected. Choose between reanalysis or PAMIP data.')
        
    # set borders
    if season == 'djf':
        ds = ds.sel(lat=slice(0,90))
    elif season == 'jas':
        ds = ds.sel(lat=slice(-90,0))
    else:
        print('Incorrect season specified.')
    ds = ds.sel(level=slice(1000,50))

    # set variables
    ubar = ds.ubar
    div1 = ds.div1_pr

    # calculate correlation using built-in Xarray function
    corr = xr.corr(div1, ubar, dim='time')
    
    #----------------------------------------------------------------------------------------------
    
    ## PLOTTING TIME
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11, 5), sharey=True)

    # ax1 LHS
    #---------
    contour = ax1.contourf(corr.lat.values, corr.level.values, corr, cmap='RdBu_r',
                        levels=np.linspace(-0.9, 0.9, 19), extend='both')
    cbar = fig.colorbar(contour, ax=ax1, location='bottom', orientation='horizontal',
                        shrink=0.75, label='correlation', extend='both', ticks=[-0.6, -0.2, 0.2, 0.6])

    # axis alterations
    ax1.invert_yaxis()
    ax1.set_xlabel('Latitude $(^\\circ N)$')
    ax1.set_yscale('linear')
    ax1.set_ylabel('Pressure (hPa)')
    ax1.set_title('$Corr(\\bar{{u}}, \\nabla_\\phi F_\\phi)$ - DJF')

    # Plot EFP box in ax1
    rect1 = patches.Rectangle((25., 600.), 50, -400, fill=False, linewidth=2)
    ax1.add_patch(rect1)

    # ax2 RHS
    #----------

    ds.ubar.mean('time').plot.contour(ax=ax2, levels=20, yincrease=False, colors='k')
    ds.div1_pr.mean('time').plot.contourf(ax=ax2, levels=20, yincrease=False,
                                            cbar_kwargs={'location': 'bottom', 'shrink': 0.55, 'label': 'DivFy'})
    ax2.set_ylabel('')

    # Plot EFP box in ax2
    rect2 = patches.Rectangle((25., 600.), 50, -400, fill=False, linewidth=2, color='limegreen')
    ax2.add_patch(rect2)

    plt.tight_layout()
    plt.show()
    
# Same as above but enables multiple plots
def plot_multiple_correlation_components(datasets_dict, data_type=None, calc_south_hemis=False):
    """
    Plot correlation components for multiple datasets in a big subplot.

    Parameters:
    - datasets_dict: Dictionary of datasets to plot (keys are dataset labels, values are datasets).
    - season: The season to consider ('djf' or 'jas').
    - reanalysis: Whether the dataset is from reanalysis.
    - pamip: Whether the dataset is from PAMIP.
    """
    
    # set different data types and the corresponding EP flux name
    data_type_mapping = {
        'reanalysis': 'div1_pr',
        'reanalysis_qg': 'div1_qg',
        'pamip': 'divF',
        'isca': 'div1'
    }
    if data_type not in data_type_mapping:
        raise ValueError(f'Invalid data_type: {data_type}. Expected one of {list(data_type_mapping.keys())}.')
    which_div1 = data_type_mapping.get(data_type)

    # Number of datasets
    N = len(datasets_dict)
    
    # Set up the figure with 2 columns per dataset
    fig, axs = plt.subplots(nrows=N, ncols=2, figsize=(10,4*N), sharey=True)
    
    # If there's only one dataset, axs will not be a 2D array. This fixes that.
    if N == 1:
        axs = [axs]
    
    # Loop over the datasets and plot for each
    for i, (label, ds) in enumerate(datasets_dict.items()):
        
        ## SET UP TIME
        
        # If required, check dimensions and variables are labelled correctly
        correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
        if not correct_dims:
            ds = data.check_dimensions(ds)
            
        # flip dimensions if required
        ds = data.check_coords(ds)
        
        # Calculate Eddy Feedback Parameter (EFP) for the dataset
        efp_value = ef.calculate_efp(ds, data_type=data_type, calc_south_hemis=calc_south_hemis)
        
        if 'ens_ax' in ds.dims:
            mean = ('time', 'ens_ax')
        else:
            mean = ('time')
        
        # Set borders
        if not calc_south_hemis:
            ds = ds.sel(lat=slice(0,90))
            season = 'djf'
        else:
            ds = ds.sel(lat=slice(-90,0))
            season = 'jas'
        ds = ds.sel(level=slice(1000,50))

        corr = ef.calculate_efp_correlation(ds, data_type=data_type, calc_south_hemis=calc_south_hemis)
        
        
        ## PLOTTING TIME

        # Left-hand subplot (correlation)
        ax1 = axs[i][0]
        contour = ax1.contourf(corr.lat.values, corr.level.values, corr, cmap='RdBu_r',
                               levels=np.linspace(-0.9, 0.9, 19), extend='both')
        cbar = fig.colorbar(contour, ax=ax1, location='bottom', orientation='horizontal',
                            shrink=0.75, label='correlation', extend='both', ticks=[-0.6, -0.2, 0.2, 0.6])

        # Axis alterations
        ax1.invert_yaxis()
        ax1.set_xlabel('Latitude $(^\\circ N)$')
        ax1.set_yscale('linear')
        ax1.set_ylabel('Pressure (hPa)')
        ax1.set_title(f'{label} ({efp_value}): $Corr(\\bar{{u}}, \\nabla_\\phi F_\\phi)$ - {season.upper()}')

        # Plot EFP box
        rect1 = patches.Rectangle((25., 600.), 50, -400, fill=False, linewidth=2)
        ax1.add_patch(rect1)

        # Right-hand subplot (mean fields)
        ax2 = axs[i][1]
        ds.ubar.mean(mean).plot.contour(ax=ax2, levels=20, yincrease=False, colors='k')
        ds[which_div1].mean(mean).plot.contourf(ax=ax2, levels=20, yincrease=False,
                                              cbar_kwargs={'location': 'bottom', 'shrink': 0.75, 'label': 'DivFy'})
        ax2.set_title(f'{label} ({efp_value}): Zonal Mean - {season.upper()}')

        # Plot EFP box
        rect2 = patches.Rectangle((25., 600.), 50, -400, fill=False, linewidth=2, color='limegreen')
        ax2.add_patch(rect2)

        # Remove Y-label on the right-hand subplots
        ax2.set_ylabel('')

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    plt.show()

    
#==================================================================================================
