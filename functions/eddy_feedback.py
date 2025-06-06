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
def calculate_epfluxes_ubar(ds, primitive=True, which_div1='divFy'):

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

    if not which_div1 in ds:
        ucomp = ds.u
        vcomp = ds.v
        temp = ds.t

        # calculate ep fluxes using aostools
        ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ucomp, vcomp, temp, do_ubar=primitive)

        # save variables to dataset
        ds['epfy'] = (ep1.dims, ep1.values)
        ds['epfz'] = (ep2.dims, ep2.values)
        ds['divFy'] = (div1.dims, div1.values)
        ds['divFz'] = (div2.dims, div2.values)

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

    # Run data through data checker
    ds = data.data_checker1000(ds)

    # define DataArray
    Fphi = ds[which_Fphi]                                                       # [m3 s-2]

    if apply_scaling:
        Fphi = pressure_scaling(Fphi, multiply_factor=multiply_factor)
        print('Scaling applied.')


    # convert lat to radians take np.cos and multiply by Fphi (inside derivative)
    lat_rads = np.deg2rad(ds.lat)
    coslat = np.cos(lat_rads)
    F_coslat = Fphi * coslat

    # calculate derivative
    deriv1 = F_coslat.differentiate('lat')                                      # [m2 s-2]

    # Divide by a cos(φ)
    a = 6.371e6
    divFphi = deriv1 / (a * coslat)                                             # [m s-2]

    # Divide by a cos(φ) AGAIN for whatever reason
    divFphi = divFphi / (a * coslat)                                            # [m s-2]

    ds[save_divFphi] = (divFphi.dims, divFphi.values)

    return ds


#==================================================================================================

#--------------------------------------
# HELPER FUNCTIONS FOR CALCULATING EFP
#--------------------------------------

def _check_data_type_divFy(ds, data_type):
    
    """
        Automatically sets the variable name for DivFy,
        depending on data type.
    """
    
    # set different data types and the corresponding EP flux name
    data_type_mapping = {
        'reanalysis': 'divFy',
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
    
    return which_div1
    
def _process_specific_efp_data(ds, data_type, season, limit_reanalysis=True):
    """
    Process dataset for specific data types and seasonal settings.

    Parameters:
    ----------
    ds : xarray.Dataset
        Input dataset to process.
    data_type : str
        Type of dataset ('reanalysis', 'reanalysis_qg', 'pamip', 'isca').
    season : str
        Season to extract (e.g., 'djf', 'jas').
    reanalysis_years : slice, optional
        Time range for reanalysis data (default: slice('1979', '2016')).

    Returns:
    -------
    xarray.Dataset
        Processed dataset.
    str
        Dimension for correlation ('time' or 'ens_ax').
    """
    corr_dim = 'time'  # Default dimension for correlation

    if data_type in ('reanalysis', 'reanalysis_qg'):
        if limit_reanalysis:
            reanalysis_years=slice('1979', '2016')
            ds = ds.sel(time=reanalysis_years)
        ds = data.seasonal_mean(ds, season=season, cut_ends=True)

    elif data_type == 'pamip':
        # Ensure time coordinate uses the correct calendar
        if not isinstance(ds.time.values[0], cftime.datetime):
            ds = ds.convert_calendar('noleap')
        
        # Handle ensembles if present
        if 'ens_ax' in ds.dims:
            ds = data.seasonal_dataset(ds, season=season)
            ds = ds.mean('time')
            corr_dim = 'ens_ax'
        else:  # Handle datasets with ensembles as separate years
            ds = data.seasonal_mean(ds, season=season)

    elif data_type == 'isca':
        ds = data.seasonal_mean(ds, season=season)

    else:
        raise ValueError(f"Unsupported data type: {data_type}. Expected 'reanalysis', 'reanalysis_qg', 'pamip', or 'isca'.")

    return ds, corr_dim

    
def _process_hemisphere(ds, calc_south_hemis):
    """
    Helper function to handle hemisphere-specific slicing and seasonal processing.

    Parameters:
    ----------
    ds : xarray.Dataset
        Dataset to process.
    calc_south_hemis : bool
        If True, process for the Southern Hemisphere.
    cut_pole : int or float
        Latitude cutoff from the pole.

    Returns:
    -------
    xarray.Dataset
        Processed dataset for the selected hemisphere and season.
    """
    
    season = 'jas' if calc_south_hemis else 'djf'
    lat_slice = slice(-90, 0) if calc_south_hemis else slice(0, 90)
    efp_lat_slice = slice(-75, -25) if calc_south_hemis else slice(25,75)

    return ds.sel(lat=lat_slice), season, efp_lat_slice


#==================================================================================================

#-----------------------------------
# CALCULATE EDDY FEEDBACK PARAMETER
#-----------------------------------


def calculate_efp(ds, data_type, calc_south_hemis=False, which_div1=None, 
                  bootstrapping=False, slice_500hPa=False):
    """
    Calculate Eddy Feedback Parameter for reanalysis and Isca data.

    Parameters:
    ----------
    ds : xarray.Dataset
        Dataset containing zonal-mean zonal wind (`ubar`) and divergence of northward EP flux (`which_div1`).
        Expected dimensions: (time, level, latitude).
    data_type : str
        Type of data ('reanalysis', 'reanalysis_qg', 'pamip', 'isca').
    calc_south_hemis : bool, optional
        If True, calculate for the Southern Hemisphere (default: False).
    take_level_mean : bool, optional
        If True, average over levels after calculations (default: True).
        If None, calculate at 500 hPa
        If False, return level array.
    reanalysis_years : slice, optional
        Years to consider for reanalysis datasets (default: slice('1979', '2016')).
    which_div1 : str, optional
        Variable name for the divergence of northward EP flux (default: 'div1_pr').

    Returns:
    -------
    float
        Eddy Feedback Parameter (rounded to 4 decimal places).
    """
    # Validate and preprocess data
    ds = data.data_checker1000(ds)
    if which_div1 != None:
        pass
    else:
        which_div1 = _check_data_type_divFy(ds, data_type=data_type)


    # Ensure required variables exist
    if not all(var in ds.variables for var in ['ubar', which_div1]):
        ds = calculate_epfluxes_ubar(ds, which_div1=which_div1)

    # Apply hemisphere-specific processing
    ds, season, efp_lat_slice = _process_hemisphere(ds, calc_south_hemis)

    # Data-specific preprocessing
    if bootstrapping:
        if data_type == 'pamip':
            ds, corr_dim = ds, 'ens_ax'
        else:
            ds, corr_dim = ds, 'time'
    else:  
        ds, corr_dim = _process_specific_efp_data(ds, data_type=data_type, season=season)
    
    #---------------------------------
    # Compute Eddy Feedback Parameter
    
    try:
        if slice_500hPa:
            ds = ds.sel(level=500., method='nearest')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr = xr.corr(ds[which_div1], ds.ubar, dim=corr_dim).load()**2

            corr = corr.sel(lat=efp_lat_slice)
            weights = np.cos(np.deg2rad(corr.lat))
            efp = corr.weighted(weights).mean('lat')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr = xr.corr(ds[which_div1], ds.ubar, dim=corr_dim).load()**2

            corr = corr.sel(lat=efp_lat_slice, level=slice(600., 200.))
            corr = corr.mean('level')

            weights = np.cos(np.deg2rad(corr.lat))
            efp = corr.weighted(weights).mean('lat')

        return round(float(efp.values), 4)
    except Exception as e:
        raise RuntimeError(f"Error during Eddy Feedback Parameter calculation: {e}")


def calculate_efp_latitude(ds, calc_south_hemis=False, which_div1='div1_pr', take_level_mean=True):
    """
    Calculate the Eddy Feedback Parameter (EFP) for specific latitudes without taking a latitude average.

    Parameters:
    ----------
    ds : xarray.Dataset
        Dataset containing `ubar` and `which_div1` variables.
        Expected dimensions: (time, level, latitude).
    calc_south_hemis : bool, optional
        If True, calculate for the Southern Hemisphere (default: False).
    cut_pole : int or float, optional
        Latitude cutoff from the pole (default: 90, full hemisphere).
    which_div1 : str, optional
        Variable name for the divergence of northward EP flux (default: 'div1').
    level_mean : bool, optional
        If True, average over levels after calculations (default: True).

    Returns:
    -------
    xarray.DataArray
        Eddy Feedback Parameter squared over the selected latitudes.
    """
    # Validate and preprocess data
    ds = data.data_checker1000(ds)

    # Apply hemisphere-specific processing
    ds, season, efp_lat_slice = _process_hemisphere(ds, calc_south_hemis)
    ds = data.seasonal_mean(ds, season=season)

    # Correlation and variance calculation
    corr = xr.corr(ds[which_div1], ds['ubar'], dim='time')
    corr = corr.sel(level=slice(600., 200.))

    if take_level_mean:
        corr = corr.mean('level')

    # Return the variance explained (EFP)
    return corr**2


def calculate_efp_correlation(ds, data_type=None, calc_south_hemis=False, which_div1='div1_pr'):
    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, level, latitude) 
    
    Output: EFP Value
    
    """
    
    # Validate and preprocess data
    ds = _check_data_type_divFy(ds, data_type=data_type)
    ds = data.data_checker1000(ds)

    # Ensure required variables exist
    if not all(var in ds.variables for var in ['ubar', which_div1]):
        ds = calculate_epfluxes_ubar(ds)

    # Apply hemisphere-specific processing
    ds, season, efp_lat_slice = _process_hemisphere(ds, calc_south_hemis)

    # Data-specific preprocessing
    ds, corr_dim = _process_specific_efp_data(ds, data_type=data_type, season=season)
    
    #---------------------------------
    # Compute Eddy Feedback Parameter CORRELATION

    # Calculate Pearson's correlation
    corr = xr.corr(ds[which_div1], ds.ubar, dim=corr_dim).load()
    
    return corr


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
