""" 
    Python file containing various functions for calculations
    surrounding the Eddy Feedback Parameter
"""
import numpy as np
import xarray as xr

import functions.aos_functions as aos
import functions.data_wrangling as data

#==================================================================================================

#----------------------
# DATASET CALCULATIONS
#----------------------


# Calculate zonal-mean zonal wind
def calculate_ubar(ds, check_variables=False, pamip_data=False):
    """
    Input: Xarray dataset
            - dim labels: (lon, ...)
            - variable: u
            
    Output: Xarray dataset with zonal-mean zonal wind 
            calculated and added as variable
    """

    # If required, check dimensions and variables are labelled correctly
    if check_variables:
        ds = data.check_dimensions(ds)
        ds = data.check_variables(ds)

    if pamip_data:
        ds['ubar'] = ds.ua.mean('lon')
    else:
        # Calculate ubar
        ds['ubar'] = ds.u.mean('lon')

    return ds

# Calculate EP fluxes
def calculate_epfluxes_ubar(ds, check_variables=False, primitive=True, pamip_data=False):

    """
    Input: Xarray dataset
            - Variables required: u,v,t
            - Dim labels: (time, lon, lat, level) 
            
            
    Output: Xarray dataset with EP fluxes calculated
            and optional calculate ubar
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    if check_variables:
        ds = data.check_dimensions(ds)
        ds = data.check_variables(ds)

    # check if ubar is in dataset also
    if not 'ubar' in ds:
        ds = calculate_ubar(ds, pamip_data=pamip_data)

    if pamip_data:
        ucomp = ds.ua
        vcomp = ds.va
        temp = ds.ta
    else:
        ucomp = ds.u
        vcomp = ds.v
        temp = ds.t

    # calculate ep fluxes using aostools
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp, do_ubar=primitive)

    # save variables to dataset
    ds['ep1'] = (ep1.dims, ep1.values)
    ds['ep2'] = (ep2.dims, ep2.values)
    ds['div1'] = (div1.dims, div1.values)
    ds['div2'] = (div2.dims, div2.values)

    # load dataset here
    ds = ds.load()

    return ds


#==================================================================================================

#-----------------------------------
# CALCULATE EDDY FEEDBACK PARAMETER
#-----------------------------------


# Calculate Eddy Feedback Parameter
def calculate_efp_reanalysis(ds, which_div1='div1_pr', take_level_mean=True, take_seasonal=True,
                             season='djf', calc_south_hemis=False, flip_latitude=False,
                             flip_level=False, check_variables=False):
    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, level, latitude) 
    
    Output: EFP Value
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    if check_variables:
        ds = data.check_dimensions(ds, ignore_dim='lon')
        ds = data.check_variables(ds)

    # flip dimensions if required
    if flip_latitude:
        # default: [-90,90]
        ds = ds.sel(lat=slice(None,None,-1))
    if flip_level:
        # default: [0,1000]
        ds = ds.sel(level=slice(None,None,-1))

    # choose hemisphere
    if calc_south_hemis:
        latitude_slice=slice(-72., -24.)
        season = 'jja'
    else:
        latitude_slice=slice(24.,72.)

    # subset dataset and take seasonal mean
    if take_seasonal:
        # take time slice to match smallest domain
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season=season)
        print('Seasonal data has been calculated for 1979-2015.')
        print()
    else:
        print('Seasonal average has not been calculated.')
        print()

    #-------------------------------------------------------------------------------

    ## CALCULATIONS

    # define variables
    ubar = ds.ubar
    div1 = ds[which_div1]

    # Calculate Pearson's correlation
    corr = xr.corr(div1, ubar, dim='time')

    # correlation squared
    corr = corr**2

    # take EFP latitude slice
    corr = corr.sel(lat=latitude_slice)

    if take_level_mean:
        corr = corr.sel(level=slice(200.,600.))
        corr = corr.mean('level')

    # Calculate weighted latitude average
    weights = np.cos( np.deg2rad(corr.lat) )
    eddy_feedback_param = corr.weighted(weights).mean('lat')

    return eddy_feedback_param.values.round(2)

# Calculate Eddy feedback parameter for PAMIP data
def calculate_efp_pamip(ds, season='djf', cut_pole=84, calc_south_hemis=False):

    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, level, latitude) 
                    - lat: [-90,90]; level: [1000,0]
    
    Output: EFP Value
    
    """

    ## CONDITIONS

    # choose hemisphere
    if calc_south_hemis:
        ds = ds.sel( lat=slice(-cut_pole, 0) )
        latitude_slice=slice(-72., -24.)
        season = 'jja'
    else:
        ds = ds.sel( lat=slice(0, cut_pole) )
        latitude_slice=slice(24.,72.)

    # Take seasonal mean
    ds = data.seasonal_dataset(ds, season=season)
    ds = ds.mean('time')

    #-------------------------------------------------------------------------------

    ## CALCULATIONS

    # Calculate Pearson's correlation
    corr = xr.corr(ds.div1, ds.ubar, dim='ens_ax')

    # correlation squared
    corr = corr**2

    # take EFP latitude slice and level mean
    corr = corr.sel( lat=latitude_slice )
    corr = corr.sel( level=slice(600., 200.) )
    corr = corr.mean('level')

    # Calculate weighted latitude average
    weights = np.cos( np.deg2rad(corr.lat) )
    eddy_feedback_param = corr.weighted(weights).mean('lat')

    return eddy_feedback_param.values.round(2)



def calculate_efp_lat_reanalysis(ds, check_vars=False, calc_south_hemis=False, cut_pole=90,
                                 which_div1='div1', take_seasonal=True, season='djf',
                                 level_mean=True, flip_level=False, flip_latitude=False):

    """ 
    Input: Xarray Dataset containing ubar and div1 
            - either div1_pr or div1_qg 
            - dims: (time, level, lat)
                - lat: [-90,90]; level: [1000,0] 

    Output: Plot showing EFP over selected latitudes
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    if check_vars:
        ds = data.check_dimensions(ds, ignore_dim='lon')
        ds = data.check_variables(ds)

    # flip dimensions if required
    if flip_latitude:
        ds = ds.sel(lat=slice(None,None,-1))
    if flip_level:
        ds = ds.sel(level=slice(None,None,-1))

    # choose hemisphere
    if calc_south_hemis:
        ds = ds.sel( lat=slice(-cut_pole, 0) )
        season = 'jja'
    else:
        ds = ds.sel( lat=slice(0, cut_pole) )
        
    # subset dataset and take seasonal mean
    if take_seasonal:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season=season)

    #----------------------------------------------------------------------

    ## Calculations
    corr = xr.corr(ds[which_div1], ds['ubar'], dim='time')
    corr = corr.sel(level=slice(600., 200.))

    if level_mean:
        corr = corr.mean('level')

    # calculate variance explained
    r = corr**2

    return r


def calculate_efp_lat_pamip(ds, season='djf', calc_south_hemis=False, cut_pole=90):

    """ 
    Input: Xarray Dataset containing PAMIP data for ubar and div1 
            - dims: (ens_ax, time, level, lat) 

    Output: Plot showing EFP over selected latitudes
    
    """

    ## CONDITIONS

    # subset dataset and take seasonal mean
    ds = data.seasonal_dataset(ds, season=season)
    ds = ds.mean('time')

    # choose hemisphere
    if calc_south_hemis:
        ds = ds.sel( lat=slice(-cut_pole, 0) )
        season = 'jja'
    else:
        ds = ds.sel( lat=slice(0, cut_pole) )

    #----------------------------------------------------------------------

    ## CALCULATIONS

    # Calculate correlation
    corr = xr.corr(ds.div1, ds['ubar'], dim='ens_ax')

    # Calculate
    corr = corr.sel(level=slice(600., 200.))
    corr = corr.mean('level')

    # calculate variance explained
    r = corr**2

    return r


#==================================================================================================

## JASMIN SERVER
if __name__ == '__main__':

    print('Opening datasets...')

    # Open datasets
    dataset = xr.open_mfdataset('/gws/nopw/j04/arctic_connect/cturrell/reanalysis_data/ \
                                                                    jra55_daily/jra55_uvtw.nc',
                            parallel=True, chunks={'time': 31})

    print('JRA55 Daily open.')

    srip = xr.open_mfdataset('/badc/srip/data/zonal/common_grid/jra_55/TEM_monthly*')
    print('SRIP Open.')

    print('Both datasets opened.')
    dataset = dataset.sel(level=srip.pressure.values)

    # calculate EP fluxes
    print('Calculating Primitive EP fluxes...')
    ds_pr = calculate_epfluxes_ubar(dataset)
    print('Calculating QG EP fluxes...')
    ds_qg = calculate_epfluxes_ubar(dataset, primitive=False)
    print('Calculations complete.')

    ds_new = xr.Dataset(data_vars={'ep1_pr': ds_pr.ep1, 'ep2_pr':ds_pr.ep2,
                                   'div1_pr':ds_pr.div1, 'div2_pr':ds_pr.div2,
                                   'ep1_qg': ds_qg.ep1, 'ep2_qg':ds_qg.ep2, 
                                   'div1_qg':ds_qg.div1, 'div2_qg':ds_qg.div2})

    # save dataset
    ds_new.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/reanalysis_data/jra55_daily/ \
                                                                            jra55_ep-fluxes.nc')


#--------------------------------------------------------------------------------------------------

## MATHS SERVERS

# if __name__ == '__main__':

#     print('Program starting...')

#     # era5
#     dataset = xr.open_mfdataset('/home/links/ct715/eddy_feedback/daily_datasets/ \
    #                                                                   era5daily_djf_uvt.nc',
#                             parallel=True, chunks={'time': 31})

#     print('Dataset has been loaded.')

#     dataset = calculate_epfluxes_ubar(dataset)

#     print('Function was successful. Now saving data...')

#     # save dataset
#     dataset.to_netcdf('/home/links/ct715/eddy_feedback/daily_datasets/era5daily_djf_uvt_ep.nc')

#     print('Dataset has been saved.')
