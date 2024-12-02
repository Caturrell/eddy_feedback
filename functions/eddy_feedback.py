""" 
    Python file containing various functions for calculations
    surrounding the Eddy Feedback Parameter
"""
# pylint: disable=invalid-name
import cftime
import warnings
import numpy as np
import xarray as xr

import functions.aos_functions as aos
import functions.data_wrangling as data

#==================================================================================================

#----------------------
# DATASET CALCULATIONS
#----------------------


# Calculate zonal-mean zonal wind
def calculate_ubar(ds):
    """
    Input: Xarray dataset
            - dim labels: (lon, ...)
            - variable: u
            
    Output: Xarray dataset with zonal-mean zonal wind 
            calculated and added as variable
    """

    # Run data through data checker
    ds = data.data_checker1000(ds)
        
    # Calculate longitudinal mean for ubar
    ds['ubar'] = ds.u.mean('lon')

    return ds

# Calculate EP fluxes
def calculate_epfluxes_ubar(ds, primitive=True):

    """
    Input: Xarray dataset
            - Variables required: u,v,t
            - Dim labels: (time, lon, lat, level) 
            
            
    Output: Xarray dataset with EP fluxes calculated
            and optional calculate ubar
    """

    ## CONDITIONS

    # Run data through data checker
    ds = data.data_checker1000(ds)


    # check if ubar is in dataset also
    if not 'ubar' in ds:
        ds = calculate_ubar(ds)

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
                      save_divFphi='divFy'):

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
        ds = data.check_dimensions(ds)

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
def calculate_efp(ds, data_type=None, calc_south_hemis=False, take_level_mean=True,
                  reanalysis_years=slice('1979', '2016'), which_div1='div1_pr'):
    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, level, latitude) 
    
    Output: EFP Value
    
    """
    
    ## DATA CHECKS
    
    # set different data types and the corresponding EP flux name
    data_type_mapping = {
        'reanalysis': which_div1,
        'reanalysis_qg': 'div1_qg',
        'pamip': None,              # handle pamip separately
        'isca': 'divFy'
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
    
    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds)
    # Check variables are named as required
    correct_vars = all(var_name in ds.variables for var_name in ['ubar', which_div1])
    if not correct_vars:
        ds = calculate_epfluxes_ubar(ds)

    # flip dimensions if required
    ds = data.check_coords(ds)
    
    #----------------------------------------------------------------------------------------------
    
    ## CONDITIONS
    
    # choose hemisphere
    if calc_south_hemis:
        latitude_slice=slice(-72., -25.)
        season = 'jas'
    elif not calc_south_hemis:
        latitude_slice=slice(25.,72.)
        season = 'djf'
        
    # variable to correlate over
    corr_dim = 'time'    
    
    # data-specific requirements
    if data_type in ('reanalysis', 'reanalysis_qg'):
        ds = ds.sel(time=reanalysis_years)
        ds = data.seasonal_mean(ds, season=season, cut_ends=True)

    elif data_type == 'pamip':
        # Convert datetime to cftime, if required
        if not isinstance(ds.time.values[0], cftime.datetime):
            ds = ds.convert_calendar('noleap')
        # Take seasonal dataset when using ensembles
        if 'ens_ax' in ds.dims:
            ds = data.seasonal_dataset(ds, season=season)
            ds = ds.mean('time')
            corr_dim='ens_ax'
        # some datasets have put all ensembles into separate years
        else:
            ds = data.seasonal_mean(ds, season=season)
    elif data_type == 'isca':
        ds = ds
    else:
        raise ValueError('Unknown data type being used.')

    #----------------------------------------------------------------------------------------------

    ## CALCULATIONS

    try:
        # Example of suppressing warnings locally
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Calculation prone to RuntimeWarning
            corr = xr.corr(ds[which_div1], ds.ubar, dim=corr_dim).load()
            corr = corr**2

        corr = corr.sel(lat=latitude_slice)
        corr = corr.sel(level=slice(600., 200.))

        if take_level_mean:
            corr = corr.mean('level')

        weights = np.cos(np.deg2rad(corr.lat))
        eddy_feedback_param = corr.weighted(weights).mean('lat')

        return eddy_feedback_param.values.round(4)
    
    except Exception as e:
        raise RuntimeError(f"An error occurred during calculation: {e}")

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
        ds = data.check_dimensions(ds)

    # flip dimensions if required
    ds = data.check_coords(ds)

    # choose hemisphere
    if calc_south_hemis:
        # take time average
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_dataset(ds, season='jas')
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

def calculate_efp_correlation(ds, data_type=None, calc_south_hemis=False, reanalysis_years=slice('1979', '2016')):
    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, level, latitude) 
    
    Output: EFP Value
    
    """
    
    ## DATA CHECKS
    
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
    
    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = data.check_dimensions(ds)
    # Check variables are named as required
    correct_vars = all(var_name in ds.variables for var_name in ['ubar', which_div1])
    if not correct_vars:
        ds = calculate_epfluxes_ubar(ds)

    # flip dimensions if required
    ds = data.check_coords(ds)
    
    #----------------------------------------------------------------------------------------------
    
    ## CONDITIONS
    
    # choose hemisphere
    if calc_south_hemis:
        latitude_slice=slice(-72., -25.)
        season = 'jas'
    elif not calc_south_hemis:
        latitude_slice=slice(25.,72.)
        season = 'djf'
        
    # variable to correlate over
    corr_dim = 'time'    
    
    # data-specific requirements
    if data_type in ('reanalysis', 'reanalysis_qg'):
        ds = ds.sel(time=reanalysis_years)
        ds = data.seasonal_mean(ds, season=season, cut_ends=True)

    elif data_type == 'pamip':
        # Convert datetime to cftime, if required
        if not isinstance(ds.time.values[0], cftime.datetime):
            ds = ds.convert_calendar('noleap')
        # Take seasonal dataset when using ensembles
        if 'ens_ax' in ds.dims:
            ds = data.seasonal_dataset(ds, season=season)
            ds = ds.mean('time')
            corr_dim='ens_ax'
        # some datasets have put all ensembles into separate years
        else:
            ds = data.seasonal_mean(ds, season=season)

    #----------------------------------------------------------------------------------------------

    ## CALCULATIONS

    # Calculate Pearson's correlation
    corr = xr.corr(ds[which_div1], ds.ubar, dim=corr_dim).load()
    
    return corr


#--------------------------------------------------------------------------------------------------

# PAMIP FUNCTIONS


# Calculate Eddy feedback parameter for PAMIP data
def calculate_efp_pamip(ds, which_div1='divF', season='djf', cut_pole=90, calc_south_hemis=False,
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
        ds = data.check_dimensions(ds)

    # choose hemisphere
    if calc_south_hemis:
        ds = ds.sel( lat=slice(-cut_pole, 0) )
        latitude_slice=slice(-72., -25.)
        season = 'jas'
    else:
        ds = ds.sel( lat=slice(0, cut_pole) )
        latitude_slice=slice(25.,72.)

    # Convert datetime to cftime, if required
    if not isinstance(ds.time.values[0], cftime.datetime):
        ds = ds.convert_calendar('noleap')

    # Take seasonal dataset when using ensembles
    if 'ens_ax' in ds.dims:
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
def calculate_efp_lat_pamip(ds, which_div1='divF', season='djf', calc_south_hemis=False,
                            usual_mean=True, cut_pole=90):

    """ 
    Input: Xarray Dataset containing PAMIP data for ubar and div1 
            - dims: (ens_ax, time, level, lat) 

    Output: Plot showing EFP over selected latitudes
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat', 'ens_ax'])
    if not correct_dims:
        ds = data.check_dimensions(ds)

    # choose hemisphere
    if calc_south_hemis:
        ds = ds.sel( lat=slice(-cut_pole, 0) )
        season='jas'
    else:
        ds = ds.sel( lat=slice(0, cut_pole) )

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

    #----------------------------------------------------------------------

    ## CALCULATIONS

    # Calculate correlation
    corr = xr.corr(ds[which_div1], ds['ubar'], dim='ens_ax')

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
