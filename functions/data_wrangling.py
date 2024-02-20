import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt

import sys

## JASMIN SERVERS
# sys.path.append('/home/users/cturrell/documents/eddy_feedback')

## MATHS SERVERS
sys.path.append('/home/links/ct715/eddy_feedback/')

import functions.aos_functions as aos 

#======================================================================================================================================

#----------------------
# DATASET MANIPULATION
#---------------------- 

# Rename dimensions in Dataset and check if ds contains Isca variable notation
def check_dimensions(ds):
    
    """
    Input: Xarray Dataset with variety of dimension labels
            - searches for 4 in particular. Look at aos function
               for more details
    
    Output: Xarray Dataset with required variable name changes
            - Checks for Isca labelling
    """
    
    # search for dimension labels
    dims = aos.FindCoordNames(ds)
    
    # rename variables usinf dict
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
    if 'ua' in ds:
        # Set renaming dict
        rename = {'ua': 'u', 'va': 'v', 'ta': 't'}
    
    # apply changes
    ds = ds.rename(rename)
    
    return ds


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
def seasonal_mean(ds):
    
    """ 
    Input: Xarray Dataset or DataArray (time, ...)
            - MUST be full year dataset
    
    Output: Xarray Dataset or DataArray with seasonal mean calculated
    
    """
    
    # resample data to start 1st Dec
    seasonal = ds.resample(time='QS-DEC').mean('time').load() 

    # take winter season of set and cut off last 'season' 
    seasonal = seasonal.sel(time=seasonal.time.dt.month.isin([12]))
    
    return seasonal 
