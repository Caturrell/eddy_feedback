""" 
    Python file containing various functions for calculations
    surrounding the Eddy Feedback Parameter
"""
# pylint: disable=invalid-name
import cftime
import numpy as np
import xarray as xr

import functions.aos_functions as aos
import functions.data_wrangling as data

#==================================================================================================

#----------------------
# DATASET CALCULATIONS
#----------------------


# Calculate zonal-mean zonal wind
def calculate_ubar(ds, pamip_data=False):
    """
    Input: Xarray dataset
            - dim labels: (lon, ...)
            - variable: u
            
    Output: Xarray dataset with zonal-mean zonal wind 
            calculated and added as variable
    """

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'lon'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    if pamip_data:
        ds['ubar'] = ds.ua.mean('lon')
    else:
        # Calculate ubar
        ds['ubar'] = ds.u.mean('lon')

    return ds

# Calculate EP fluxes
def calculate_epfluxes_ubar(ds, primitive=True, pamip_data=False):

    """
    Input: Xarray dataset
            - Variables required: u,v,t
            - Dim labels: (time, lon, lat, level) 
            
            
    Output: Xarray dataset with EP fluxes calculated
            and optional calculate ubar
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'lon'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

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

# Rescale DataArray for log-pressure to pressure or visa versa
def pressure_scaling(da, p0=1e3, multiply_factor=True):

    """
        Converts log-pressure coordinates to pressure coordinates
        or visa versa.
        ----------------------------------------------------------
        
        Input: xr.DataArray in log-pressure/pressure coordinates
                - (pressure, latitude)
        
        Output: xr.DataArray in pressure/log-pressure coordinates
                - (pressure, latitude)
    """

    # define dimensions
    lat = da.lat.values
    level = da.level.values

    # define and calculate ratio
    p_ratio = np.repeat(level/p0, lat.size).reshape((level.size,lat.size))

    if multiply_factor:
        da_new = da * p_ratio
    else:
        da_new = da / p_ratio

    return da_new

# Calculate divergence of northward component of EP flux
def calculate_divFphi(ds, which_Fphi='epfy', apply_scaling=False, multiply_factor=True,
                      save_divFphi='divF'):

    """
        Calculate divergence of northward component
        of EP flux, F_phi. Including an optional 
        scaling from log-pressure to pressure coords.
        
        ----------------------------------------------
        
        Input: xr.Dataset containing epfy/Fy/Fphi [m3 s-2]
        
        Out: xr.DataArray Div_Fphi
    """

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds, ignore_dim='lon')

    # define DataArray
    Fphi = ds[which_Fphi]                                                       # [m3 s-2]

    if apply_scaling:
        Fphi = pressure_scaling(Fphi, multiply_factor=multiply_factor)
        print('Scaling applied.')


    # convert lat to radians take np.cos and multiply by Fphi (inside derivative)
    lat_rads = np.deg2rad(ds.lat.values)
    coslat = np.cos(lat_rads)
    F_coslat = Fphi * coslat

    # calc derivative and convert lat dimension to radians
    F_coslat['lat'] = lat_rads
    deriv1 = F_coslat.differentiate('lat')                                      # [m2 s-2]

    # Divide by a cos(φ)
    a = 6.371e6
    divFphi = deriv1 / (a * coslat)                                             # [m s-2]

    # Divide by a cos(φ) AGAIN for whatever reason
    divFphi = divFphi / (a * coslat)                                            # [m s-2]

    ds[save_divFphi] = (divFphi.dims, divFphi.values)

    return ds


#==================================================================================================

#-----------------------------------
# CALCULATE EDDY FEEDBACK PARAMETER
#-----------------------------------


# Calculate Eddy Feedback Parameter for reanalysis and Isca data
def calculate_efp(ds, which_div1='div1_pr', take_level_mean=True, calc_south_hemis=False):
    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, level, latitude) 
    
    Output: EFP Value
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds, ignore_dim='lon')

    # flip dimensions if required
    if (ds.level[0] - ds.level[1]) < 0:
        # default: [1000,0]
        ds = ds.sel(level=slice(None,None,-1))
    if (ds.lat[0] + ds.lat[1]) > 0:
        # default: [-90,90]
        ds = ds.sel(lat=slice(None,None,-1))

    # choose hemisphere
    if calc_south_hemis:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_dataset(ds, season='jas')
        latitude_slice=slice(-72., -24.)
    else:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season='djf')
        latitude_slice=slice(24.,72.)

    #-------------------------------------------------------------

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
        corr = corr.sel(level=slice(600., 200.))
        corr = corr.mean('level')

    # Calculate weighted latitude average
    weights = np.cos( np.deg2rad(corr.lat) )
    eddy_feedback_param = corr.weighted(weights).mean('lat')

    return eddy_feedback_param.values.round(4)

# Calculate EFP without taking the latitude average
def calculate_efp_latitude(ds, calc_south_hemis=False, cut_pole=90,
                                which_div1='div1', level_mean=True):

    """ 
    Input: Xarray Dataset containing ubar and div1 
            - either div1_pr or div1_qg 
            - dims: (time, level, lat)
                - lat: [-90,90]; level: [1000,0] 

    Output: Plot showing EFP over selected latitudes
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds, ignore_dim='lon')

    # flip dimensions if required
    if (ds.level[0] - ds.level[1]) < 0:
        # default: [1000,0]
        ds = ds.sel(level=slice(None,None,-1))
    if (ds.lat[0] + ds.lat[1]) > 0:
        # default: [-90,90]
        ds = ds.sel(lat=slice(None,None,-1))

    # choose hemisphere
    if calc_south_hemis:
        # take time average
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_dataset(ds, season='jas')
        ds = ds.mean('time')
        # take lat slice
        ds = ds.sel( lat=slice(-cut_pole, 0) )
    else:
        # take time avg
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season='djf')
        # take lat slice
        ds = ds.sel( lat=slice(0, cut_pole) )

    #----------------------------------------------------------------------

    ## Calculations
    corr = xr.corr(ds[which_div1], ds['ubar'], dim='time')
    corr = corr.sel(level=slice(600., 200.))

    if level_mean:
        corr = corr.mean('level')

    # calculate variance explained
    r = corr**2

    return r


#--------------------------------------------------------------------------------------------------

# PAMIP FUNCTIONS


# Calculate Eddy feedback parameter for PAMIP data
def calculate_efp_pamip(ds, which_div1='divF', season='djf', cut_pole=84, calc_south_hemis=False,
                        usual_mean=True):

    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (time, level, lat) 
                    - lat: [-90,90]; level: [1000,0]
    
    Output: EFP Value
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'ens_ax'])
    if not correct_dims:
        ds = data.check_dimensions(ds, ignore_dim='lon')

    # choose hemisphere
    if calc_south_hemis:
        ds = ds.sel( lat=slice(-cut_pole, 0) )
        latitude_slice=slice(-72., -24.)
        season = 'jas'
    else:
        ds = ds.sel( lat=slice(0, cut_pole) )
        latitude_slice=slice(24.,72.)

    # Convert datetime to cftime, if required
    if not isinstance(ds.time.values[0], cftime.datetime):
        ds = ds.convert_calendar('noleap')

    # Take seasonal dataset when using ensembles
    if usual_mean:
        ds = data.seasonal_dataset(ds, season=season)
        ds = ds.mean('time')
    # some datasets have put all ensembles into separate years
    else:
        if calc_south_hemis:
            ds = data.seasonal_dataset(ds, season='jas')
            ds = ds.groupby('time.year').mean('time')
            ds = ds.rename({'year': 'ens_ax'})
        else:
            ds = data.seasonal_mean(ds, season=season, cut_ends=False)
            ds = ds.rename({'time': 'ens_ax'})

    #-------------------------------------------------------------------------------

    ## CALCULATIONS

    # Calculate Pearson's correlation
    corr = xr.corr(ds[which_div1], ds.ubar, dim='ens_ax').load()

    # correlation squared
    corr = corr**2

    # take EFP latitude slice and level mean
    corr = corr.sel( lat=latitude_slice )
    corr = corr.sel( level=slice(600., 200.) )
    corr = corr.mean('level')

    # Calculate weighted latitude average
    weights = np.cos( np.deg2rad(corr.lat) )
    eddy_feedback_param = corr.weighted(weights).mean('lat')

    eddy_feedback_param.load()

    return eddy_feedback_param.values.round(4)


# Calculate EFP for PAMIP data without taking latitudinal average
def calculate_efp_lat_pamip(ds, season='djf', calc_south_hemis=False, cut_pole=90):

    """ 
    Input: Xarray Dataset containing PAMIP data for ubar and div1 
            - dims: (ens_ax, time, level, lat) 

    Output: Plot showing EFP over selected latitudes
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'ens_ax'])
    if not correct_dims:
        ds = data.check_dimensions(ds, ignore_dim='lon')

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
