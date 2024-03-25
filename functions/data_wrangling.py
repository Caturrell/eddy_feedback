import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt

import aos_functions as aos 

#======================================================================================================================================

#----------------------
# DATASET MANIPULATION
#---------------------- 

# Rename dimensions in Dataset and check if ds contains Isca variable notation
def check_dimensions(ds, ignore_dim=None):
    
    """
    Input: Xarray Dataset with variety of dimension labels
            - searches for 4 in particular. Look at aos function
               for more details
    
    Output: Xarray Dataset with required variable name changes
            - Checks for Isca labelling
    """
    
    # search for dimension labels
    dims = aos.FindCoordNames(ds)
    

    # rename variables using dict
    if ignore_dim == 'lon':
        rename = {dims['lat']: 'lat', dims['pres']: 'level'}
    else:
        rename = {dims['lon']: 'lon', dims['lat']: 'lat', dims['pres']: 'level'} 

    
    # rename dataset
    ds = ds.rename(rename)
    
    return ds


# Rename dimensions in Isca to suit my function needs
def check_variables(ds):
    
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
    elif 'ua' in ds:
        # Set renaming dict
        rename = {'ua': 'u', 'va': 'v', 'ta': 't'}
    else:
        rename = {}

    
    # apply changes
    ds = ds.rename(rename)
    
    return ds


# # flip coordinates
# def flip_coords(ds, dim=None):

#     """ 
#     Input: Xarray Dataset
#             - Any dims

#     Output: Equivalent Xarray Dataset with flipped
#             chosen coordinate
    
#     """

#     ds = ds[dim].sel()



#======================================================================================================================================

#----------------------
# DATASET SUBSETTING
#---------------------- 


# Subset to seasonal datasets
def seasonal_dataset(ds, season='djf', save_ds=False, save_location='./ds.nc'):
    
    """ 
    Input: Xarray dataset for full year
    
    Output: Xarray dataset with required season data 
    
    """
    
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

# Calculate annual means
def annual_mean(ds):
    
    """ 
    Input: Xarray Dataset or DataArray (time, ...)
    
    Output: Xarray Dataset or DataArray with annual mean calculated
    
    """
    
    # calculate annual mean
    ds = ds.groupby('time.year').mean('time').load()
    
    return ds 

# Calculate annual means
def seasonal_mean(ds, cut_ends=True, season=None):
    
    """ 
    Input: Xarray Dataset or DataArray (time, ...)
            - MUST be full year dataset
    
    Output: Xarray Dataset or DataArray with seasonal mean calculated
    
    """

    # remove first Jan and Feb, and final Dec to ensure FULL seasons
    if cut_ends:
        # slice data from first March to final November
        # (assuming dataset is JanYYYY - DecYYYY) 
        ds = ds.sel(time=slice(f'{ds.time.dt.year[0].values}-3', f'{ds.time.dt.year[-1].values}-11'))

    
    # resample data to start 1st Dec
    seasonal = ds.resample(time='QS-DEC').mean('time').load() 

    # take winter season of set and cut off last 'season' 
    if season == 'djf':
        seasonal = seasonal.sel(time=seasonal.time.dt.month.isin([12]))
    elif season =='mam':
        seasonal = seasonal.sel(time=seasonal.time.dt.month.isin([3]))
    elif season =='jja':
        seasonal = seasonal.sel(time=seasonal.time.dt.month.isin([6]))
    elif season =='son':
        seasonal = seasonal.sel(time=seasonal.time.dt.month.isin([9]))
    
    return seasonal 
