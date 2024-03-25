import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import aos_functions as aos 
import data_wrangling as data 


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
def calculate_efp(ds, which_div1='div1_pr', take_level_mean=True, take_seasonal=True, season='djf',
                  calculate_SH=False, flip_latitude=False, flip_level=False, check_variables=False): 
    
    """ 
    Input: Xarray DataSet containing zonal-mean zonal wind (ubar)
            and divergence of northward EP flux (div1)
                - dims: (year, latitude) 
    
    Output: EFP Value at chosen level slice 
    
    """ 

    ## CONDITIONS

    # If required, check dimensions and variables are labelled correctly
    if check_variables:
        ds = data.check_dimensions(ds, ignore_dim='lon')
        ds = data.check_variables(ds) 

    # flip dimensions if required
    if flip_latitude:
        ds = ds.sel(lat=slice(None,None,-1))
    if flip_level:
        ds = ds.sel(level=slice(None,None,-1))

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
    if calculate_SH:
        corr = corr.sel(lat=slice(-75., -25.))
    else:
        corr = corr.sel(lat=slice(25.,72.)) 

    
    if take_level_mean:
        corr = corr.sel(level=slice(200.,600.))
        corr = corr.mean('level')

    # Calculate weighted latitude average 
    weights = np.cos( np.deg2rad(corr.lat) )
    EFP = corr.weighted(weights).mean('lat') 

    return EFP 



def calculate_efp_latitude(ds, check_variables=False, latitude='NH', which_div1='div1_pr', 
                           take_seasonal=True, level_mean=True, flip_level=False, flip_latitude=False):
    
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

    # flip dimensions if required
    if flip_latitude:
        ds = ds.sel(lat=slice(None,None,-1))
    if flip_level:
        ds = ds.sel(level=slice(None,None,-1))

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
    corr = corr.sel(level=slice(200.,600.))

    if level_mean:
        corr = corr.mean('level')
    

    # calculate variance explained
    r = corr**2

    return r 


    

#===================================================================================================================

## JASMIN SERVER
if __name__ == '__main__':
    
    print('Opening datasets...')
    
    # Open datasets
    ds = xr.open_mfdataset('/gws/nopw/j04/arctic_connect/cturrell/reanalysis_data/jra55_daily/jra55_uvtw.nc', 
                            parallel=True, chunks={'time': 31})
    
    print('JRA55 Daily open.')
    
    srip = xr.open_mfdataset('/badc/srip/data/zonal/common_grid/jra_55/TEM_monthly*')
    print('SRIP Open.')

    print('Both datasets opened.')
    ds = ds.sel(level=srip.pressure.values)
    
    # calculate EP fluxes
    print('Calculating Primitive EP fluxes...')
    ds_pr = calculate_epfluxes_ubar(ds) 
    print('Calculating QG EP fluxes...')
    ds_qg = calculate_epfluxes_ubar(ds, primitive=False)
    print('Calculations complete.')
    
    ds_new = xr.Dataset(data_vars={'ep1_pr': ds_pr.ep1, 'ep2_pr':ds_pr.ep2, 'div1_pr':ds_pr.div1, 'div2_pr':ds_pr.div2,
                                   'ep1_qg': ds_qg.ep1, 'ep2_qg':ds_qg.ep2, 'div1_qg':ds_qg.div1, 'div2_qg':ds_qg.div2})
    
    # save dataset
    ds_new.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/reanalysis_data/jra55_daily/jra55_ep-fluxes.nc')
    

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