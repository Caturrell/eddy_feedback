"""
    A script containing functions for plotting some wave number plots
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xarray as xr

import functions.data_wrangling as data

# Plot components
def plot_wavenumber_EFP_components(ax, ds, data_type=None, which_div1='div1_pr', hemisphere=None, TOA=10.,
                        title=''):
    """
    Input: DataArrays of ubar and F_phi
            - Dims: (time, level, lat)
                - DATASET MUST BE FULL YEAR FOR SEASONAL MEAN 
            - Subsetted to NH or SH
                - Cut off stratosphere >10. hPa

    Output: Plot components of EFP.
    """

    ## DATA CHECKS
    
    # set different data types and the corresponding EP flux name
    data_type_mapping = {
        'reanalysis': which_div1,
        'reanalysis_qg': 'div1_qg',
        'pamip': None,              # handle pamip separately
        'isca': 'div1'
    }
    if data_type not in data_type_mapping:
        raise ValueError(f'Invalid data_type: {data_type}. Expected one of {list(data_type_mapping.keys())}.')
    
    # catch other pamip definition
    # Check if 'divF' or 'divFy' exists in the dataset and set which_div1
    if data_type == 'pamip':
        if 'divF' in ds.variables:
            which_div1 = 'divF'
        elif 'divFy' in ds.variables:
            which_div1 = 'divFy'
        else:
            raise ValueError("Neither 'divF' nor 'divFy' found in dataset for pamip data type.")
    else:
        which_div1 = data_type_mapping.get(data_type)
        
    ds = data.data_checker1000(ds)
        
    if hemisphere == 'NH':
        ds = data.seasonal_mean(ds, season='djf')
        ds = ds.sel(lat=slice(0,90))
    elif hemisphere == 'SH':
        ds = data.seasonal_mean(ds, season='jas')
        ds = ds.sel(lat=slice(-90,0))
    else:
        print('Hemisphere not specified.')
        
    if TOA != None:
        ds = ds.where(ds.level > 10.)
        
    #-------------------------------------------------------------------

    # Plotting components
    # ax.contour(ds.lat, ds.level, ds.ubar.mean('time'), levels=20, yincrease=False, colors='k')
    # ax.contourf(ds.lat, ds.level, ds[which_div1].mean('time'), levels=20, yincrease=False)
    
    ds.ubar.mean('time').plot.contour(levels=20, colors='k', ax=ax)
    
    # colour_levels=np.linspace(-5e-5, 5e-5, 21)
    colour_levels=21
    contourf = ds[which_div1].mean('time').plot.contourf(levels=colour_levels, ax=ax, add_colorbar=False)

    # Plot EFP box
    rect = patches.Rectangle((25., 600.), 50, -400, fill=False, linewidth=2, color='limegreen')
    ax.add_patch(rect)

    ax.set_title(title)
    ax.invert_yaxis()

    return contourf

# Plot correlations
def plot_wavenumber_EFP_correlation(ax, ds, data_type, logscale=True, show_rect=True, hemisphere='NH',
                                top_atmos=10., cut_poles=False, title='', take_seasonal=True,
                                which_div1='div1_pr'):
    """
    Input: DataArrays of ubar and F_phi
            - Dims: (time, level, lat)
                - DATASET MUST BE FULL YEAR FOR SEASONAL MEAN 
            - Subsetted to NH or SH
            - Cut off stratosphere >10. hPa
    
    Output: contourf plot matching Fig.6a in Smith et al., 2022 
    """

    ## DATA CHECKS
    
    # set different data types and the corresponding EP flux name
    data_type_mapping = {
        'reanalysis': which_div1,
        'reanalysis_qg': 'div1_qg',
        'pamip': None,              # handle pamip separately
        'isca': 'div1'
    }
    if data_type not in data_type_mapping:
        raise ValueError(f'Invalid data_type: {data_type}. Expected one of {list(data_type_mapping.keys())}.')
    
    # catch other pamip definition
    # Check if 'divF' or 'divFy' exists in the dataset and set which_div1
    if data_type == 'pamip':
        if 'divF' in ds.variables:
            which_div1 = 'divF'
        elif 'divFy' in ds.variables:
            which_div1 = 'divFy'
        else:
            raise ValueError("Neither 'divF' nor 'divFy' found in dataset for pamip data type.")
    else:
        which_div1 = data_type_mapping.get(data_type)
        
    ds = data.data_checker1000(ds)

    # choose hemisphere
    if  hemisphere == 'NH':
        ds = ds.where(ds.lat >= 0., drop=True)
        rect_box = (25., 600.)
        season='djf'
    elif  hemisphere == 'SH':
        ds = ds.where(ds.lat <= 0., drop=True)
        rect_box = (-75., 600.)
        season='jas'
    else:
        print('Hemisphere not specified')

    # separate time into annual means
    # and use .load() to force the calculation now
    if take_seasonal:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season=season)


    # set variables
    ubar = ds.ubar
    div1 = ds[which_div1]

    # suppress RuntimeWarning temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # calculate correlation using built-in Xarray function
        corr = xr.corr(div1, ubar, dim='time').load()

    # choose top of atmosphere
    corr = corr.where(corr.level >= top_atmos, drop=True)

    if cut_poles:
        corr = corr.where(corr.lat >= -85., drop=True)
        corr = corr.where(corr.lat <= 85., drop=True)

    #------------------------------------------------------------------

    ## PLOTTING TIME

    # actual plotting
    contour = ax.contourf(corr.lat.values, corr.level.values, corr, cmap='RdBu_r',
                          levels=np.linspace(-0.9, 0.9, 19), extend='both')
    
    cbar = plt.colorbar(contour, ax=ax, location='bottom', orientation='horizontal', shrink=0.75,
                        label='correlation', extend='both', ticks=[-0.6, -0.2, 0.2, 0.6])

    # axis alterations
    ax.invert_yaxis()
    ax.set_xlabel(' $(^\\circ N)$')
    if logscale:
        ax.set_yscale('log')
        ax.set_ylabel('Log pressure (hPa)')
    else:
        ax.set_ylabel('Pressure (hPa)')
    
    ax.set_title(f'{title}')

    # Plot EFP box
    if show_rect:
        rect = patches.Rectangle(rect_box, 50, -400,
                                  fill=False, linewidth=2)
        ax.add_patch(rect)




