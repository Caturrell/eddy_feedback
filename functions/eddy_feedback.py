import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import functions.aos_functions as aos 
import functions.data_wrangling as data 


#======================================================================================================================================

#----------------------
# DATASET CALCULATIONS
#----------------------


# Calculate zonal-mean zonal wind
def calculate_ubar(ds, check_variables=False):
    
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


    # Calculate ubar
    ds['ubar'] = ds.u.mean('lon')
    
    return ds

# Calculate EP fluxes
def calculate_epfluxes_ubar(ds, check_variables=False, primitive=True):
    
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
        ds = calculate_ubar(ds) 
        
    # calculate ep fluxes using aostools
    ep1, ep2, div1, div2 = aos.ComputeEPfluxDivXr(ds.u, ds.v, ds.t, do_ubar=primitive)

    # save variables to dataset
    ds['ep1'] = (ep1.dims, ep1.values)
    ds['ep2'] = (ep2.dims, ep2.values)
    ds['div1'] = (div1.dims, div1.values)
    ds['div2'] = (div2.dims, div2.values)
    
    return ds
        
    
# Calculate Eddy Feedback Parameter
def calculate_efp_sliced(ds, which_div1='div1_pr', lat_slice=slice(72.,25.),
                         check_variables=False, take_seasonal=True): 
    
    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, latitude) 
                - sliced at selected level (hPa)
    
    Output: EFP Value at chosen level slice 
    
    """ 

    # If required, check dimensions and variables are labelled correctly
    if check_variables:
        ds = data.check_dimensions(ds, ignore_dim='lon')
        ds = data.check_variables(ds) 

    # subset dataset and take seasonal mean
    if take_seasonal:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season='djf')

    # define variables
    ubar = ds.ubar
    div1 = ds[which_div1]
    
    # Calculate Pearson's correlation
    r = xr.corr(div1, ubar, dim='time')

    # correlation squared
    r = r**2
    
    # take EFP latitude slice if required
    r = r.sel(lat=lat_slice)
    
    # Calculate weighted latitude average 
    weights = np.cos( np.deg2rad(r.lat) )
    EFP = r.weighted(weights).mean('lat') 
    
    return EFP 



def calculate_efp_latitude(ds, check_variables=False, latitude='NH', which_div1='div1_pr', 
                           take_seasonal=True, level_mean=True):
    
    """ 
    Input: Xarray Dataset containing ubar and div1 
            - either div1_pr or div1_qg 
            - dims: (time, level, lat) 

    Output: Plot showing EFP over selected latitudes
    
    """

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    if check_variables:
        ds = data.check_dimensions(ds, ignore_dim='lon')
        ds = data.check_variables(ds)  

    # subset dataset and take seasonal mean
    if take_seasonal:
        ds = ds.sel(time=slice('1979', '2016'))
        ds = data.seasonal_mean(ds, season='djf')

    # select latitude
    if latitude == 'NH':
        lat_slice = slice(0,90)
    elif latitude == 'SH':
        lat_slice = slice(-90,0) 

    #----------------------------------------------------------------------
        
    ## Calculations
    corr = xr.corr(ds[which_div1], ds['ubar'], dim='time')
    corr = corr.sel(lat=lat_slice)
    corr = corr.sel(level=slice(600., 200.))

    if level_mean:
        corr = corr.mean('level')
    

    # calculate variance explained
    r = corr**2

    return r 


    

#===================================================================================================================

## JASMIN SERVER
# if __name__ == '__main__':
    
#     # era5
#     ds = xr.open_mfdataset('/gws/nopw/j04/arctic_connect/cturrell/era5_data/era5daily_djf_uvt.nc', 
#                             parallel=True, chunks={'time': 31})
    
#     ds = calculate_epfluxes_ubar(ds) 
    
#     # save dataset
#     ds.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/era5_data/era5daily_djf_uvt_ep.nc')
    

#-------------------------------------------------------------------------------------------------------------------
    

## MATHS SERVERS
    
# if __name__ == '__main__':

#     print('Program starting...')
    
#     # era5
#     ds = xr.open_mfdataset('/home/links/ct715/eddy_feedback/daily_datasets/era5daily_djf_uvt.nc', 
#                             parallel=True, chunks={'time': 31}) 
    
#     print('Dataset has been loaded.')
    
#     ds = calculate_epfluxes_ubar(ds) 

#     print('Function was successful. Now saving data...') 
    
#     # save dataset
#     ds.to_netcdf('/home/links/ct715/eddy_feedback/daily_datasets/era5daily_djf_uvt_ep.nc')

#     print('Dataset has been saved.')