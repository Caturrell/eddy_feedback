"""
    Python file containing various data wrangling functions.
"""
# pylint: disable=line-too-long
# pylint: disable=wrong-import-position
# import os
import sys
import warnings
import cftime
import datetime
import numpy as np
import xarray as xr
# import xesmf as xe

sys.path.append('/home/users/cturrell/documents/eddy_feedback')
import functions.aos_functions as aos
import functions.eddy_feedback as ef

#==================================================================================================

#----------------------
# DATASET MANIPULATION
#----------------------

# Rename dimensions in Dataset and check if ds contains Isca variable notation
def check_dimensions(ds):
    """
    Input: Xarray Dataset with variety of dimension labels
            - searches for 4 in particular (lon, lat, pres, time).
            - Uses aos.FindCoordNames to get dimension labels.
    
    Output: Xarray Dataset with required dimension name changes.
            - Renames Isca labels to standard names.
    """

    # search for dimension labels
    dims = aos.FindCoordNames(ds)
    
    if 't' in ds.dims:
        ds = ds.rename({'t': 'time'})
    elif 'record' in ds.dims:
        ds = ds.rename({'record': 'ens_ax'})

    # base renaming dictionary
    try:
        rename_dict = {
            dims['lon']: 'lon',
            dims['lat']: 'lat',
            dims['pres']: 'level'
        }
    except KeyError:
        rename_dict = {
            dims['lat']: 'lat',
            dims['pres']: 'level'
        }

    # Apply renaming
    return ds.rename(rename_dict)


# Rename dimensions in Isca to suit my function needs
def check_variables(ds):
    """
    Input: Xarray Dataset from Isca or PAMIP simulation
            - dimensions: (time, lon, lat, pfull) or (time, lon, lat, level)
            - variables: ucomp, vcomp, temp (Isca) or ua, va, ta (PAMIP)
    
    Output: Xarray DataSet with renamed dimensions and variables
            - dimensions: (time, lon, lat, level)
            - variables: u, v, t
    """

    rename_dict = {
        'ucomp': 'u',  # Isca u component
        'vcomp': 'v',  # Isca v component
        'temp': 't',   # Isca temperature
        'ua': 'u',     # PAMIP u component
        'va': 'v',     # PAMIP v component
        'ta': 't'      # PAMIP temperature
    }

    # Apply renaming for variables found in the dataset
    return ds.rename({k: v for k, v in rename_dict.items() if k in ds})

def check_coords(ds):
    
    """
    
    Check coordinates are in correct orientation
    
    Input: Xarray Dataset
            - dimensions: (time, lon, lat, pfull)
            - variables: ucomp, vcomp, temp
    
    Output: Xarray DataSet with required name changes
            - dimensions: (time, lon, lat, level)
            - variables: u,v,t
    """
    
    # Check pressure coords are [1000,0]
    if ds.level.max() > 1000.00:
        ds['level'] = ds['level'] / 100
    
    # flip dimensions if required
    if (ds.level[0] - ds.level[1]) < 0:
        # default: [1000,0]
        ds = ds.sel(level=slice(None,None,-1))
    if (ds.lat[0] + ds.lat[1]) > 0:
        # default: [-90,90]
        ds = ds.sel(lat=slice(None,None,-1))
        
    return ds

def data_checker1000(ds, check_vars=True):
    
    """
        Function that runs through all the above checks
    """
    
    # Check dimensions are labelled in my convention
    ds = check_dimensions(ds)
        
    # check coordinates are correct orientation
    ds = check_coords(ds)
    
    if check_vars:
        ds = check_variables(ds)
    
    return ds



#==================================================================================================

#----------------------
# DATASET SUBSETTING
#----------------------

def seasonal_dataset(ds, season='djf', save_ds=False, save_location='./ds.nc'):
    """
    Subset an Xarray dataset to the specified season.

    Input: 
    - ds: Xarray dataset for the full year.
    - season: Season to extract ('djf', 'mam', 'jja', 'son', 'jas').
    - save_ds: Whether to save the output dataset (default: False).
    - save_location: Location to save the dataset if save_ds is True (default: './ds.nc').

    Output: Xarray dataset with the required season data.
    
    """
    # Define season-month mappings
    season_months = {
        'djf': [12, 1, 2],
        'mam': [3, 4, 5],
        'jja': [6, 7, 8],
        'son': [9, 10, 11],
        'jas': [7, 8, 9]  # PAMIP SH summer
    }

    if season in season_months:
        ds = ds.sel(time=ds.time.dt.month.isin(season_months[season]))
    else:
        raise ValueError(f'Invalid input. Season {season} is not a valid option.')

    if save_ds:
        ds.to_netcdf(save_location)

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


# Calculate seasonal means
def seasonal_mean(ds, cut_ends=False, season=None):
    """
    Calculate seasonal means for an Xarray dataset.

    Input: 
    - ds: Xarray Dataset or DataArray (must be a full-year dataset).
    - cut_ends: If True, removes incomplete seasonal data at the start and end of the dataset.
    - season: Season to compute the mean for ('djf', 'mam', 'jja', 'son', 'jas').

    Output: Xarray Dataset or DataArray with seasonal means calculated.
    
    """
    
    if season == None:
        raise ValueError(f'Invalid season: {season}. Choose valid seasonal months.')
    
    # Define required dimensions that should always be present
    required_dims = ['time', 'lat']

    # If 'level' is present in the dataset, include it in the check
    if 'level' in ds.dims:
        required_dims.append('level')

    # Check and fix dimensions if necessary
    if not all(dim in ds.dims for dim in required_dims):
        ds = check_dimensions(ds)


    # Remove incomplete seasons if cut_ends is True
    if cut_ends:
        start_year = ds.time.dt.year[0].values
        end_year = ds.time.dt.year[-1].values
        ds = ds.sel(time=slice(f'{start_year}-03', f'{end_year}-11'))

    # Season-specific means
    if season == 'jas':
        seasonal = ds.sel(time=ds.time.dt.month.isin([7, 8, 9]))
    else:
        seasonal = ds.resample(time='QS-DEC').mean('time')

    # Further selection of specific months for certain seasons
    season_months = {
        'djf': 12,
        'mam': 3,
        'jja': 6,
        'son': 9
    }

    if season in season_months:
        seasonal = seasonal.sel(time=seasonal.time.dt.month == season_months[season])

    if season == 'jas':
        seasonal = seasonal.groupby('time.year').mean('time').rename({'year': 'time'})

    return seasonal



#==================================================================================================

#----------------------
# DATASET PROCESSING
#----------------------

# change time for pamip data
def change_to_cftime(da):
    """ Takes DataArray and converts time to a common cftime """
    
    # sort out time by converting required datasets to cftime.NoLeap
    # and ensure both vars have same date and time for monthly data
    da = da.convert_calendar('noleap')
    
    # # Extract the time component
    da_times = da['time'].values

    # Function to change a date to the first of the month
    def to_first_of_month(date):
        return cftime.DatetimeNoLeap(date.year, date.month, 1, 0, 0, 0, 0, calendar=date.calendar)

    # Apply the function to all dates
    new_times = [to_first_of_month(t) for t in da_times]

    # Replace the original time component with the new times
    da['time'] = new_times
    
    return da

# # regrid PAMIP data
# def regrid_dataset_3x3(ds, check_dims=False):
#     """
#     Input: Xarray dataset
#             - Must contain (lat, lon)
            
#     Output: Regridded Xarray dataset 
#             to 3 deg lat lon
    
#     """
#     # Rename
#     if check_dims:
#         ds = check_dimensions(ds)
#         ds = check_variables(ds)
#     # build regridder
#     ds_out = xr.Dataset(
#         {
#             'lat': (['lat'], np.arange(-90, 93, 3)),
#             'lon': (['lon'], np.arange(0,360, 3))
#         }
#     )

#     regridder = xe.Regridder(ds, ds_out, "bilinear")
#     ds_new = regridder(ds)

#     # verify that the result is the same as regridding each variable one-by-one
#     for k in ds.data_vars:
#         print(k, ds_new[k].equals(regridder(ds[k])))

#     print('Regridding and checks complete. Dataset ready.')
#     return ds_new

# # regrid data to chosen degree
# def regrid_dataset(ds, deg=3, check_dims=False):
#     """
#     Input: Xarray dataset
#             - Must contain (lat, lon)
            
#     Output: Regridded Xarray dataset 
#             to 3 deg lat lon
    
#     """
#     # Rename
#     if check_dims:
#         ds = check_dimensions(ds)
#         ds = check_variables(ds)
#     # build regridder
#     ds_out = xr.Dataset(
#         {
#             'lat': (['lat'], np.arange(-90, 90+deg, deg)),
#             'lon': (['lon'], np.arange(0,360, deg))
#         }
#     )

#     regridder = xe.Regridder(ds, ds_out, "bilinear")
#     ds_new = regridder(ds)

#     # verify that the result is the same as regridding each variable one-by-one
#     for k in ds.data_vars:
#         print(k, ds_new[k].equals(regridder(ds[k])))

#     print('Regridding and checks complete. Dataset ready.')
#     return ds_new



def process_pamip_monthly(model=None):
    """ 
        Process monthly PAMIP data to be regridded to 3x3
        and subset the dataset to match SRIP NaNs.
        Also calculates divFy.
        ---------------------------------------------------
        Input: Model name
        
        Output: Saved dataset with divFy
    """

    # suppress serialisation warning
    if model == 'CESM2':
        warnings.filterwarnings('ignore', category=xr.SerializationWarning)

    # import datasets
    epfy = xr.open_mfdataset(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/1.6_pdSST-futArcSIC/epfy/{model}/*.nc',
                            combine='nested', concat_dim='ens_ax', parallel=True)
    ua = xr.open_mfdataset(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/1.6_pdSST-futArcSIC/ua/{model}/*.nc',
                            combine='nested', concat_dim='ens_ax', parallel=True)

    ## Model specific needs:
    if model == 'EC-EARTH3':
        epfy = epfy.mean('lon')
    if model == 'CNRM-CM6-1':
        epfy = epfy.mean('lon')
    if model == 'HadGEM3-GC31-LL':
        ua = ua.rename({'u': 'ua', 't': 'time', 'p_1': 'level', 'latitude_1': 'lat', 'longitude_1': 'lon'})
        ua['level'] = ua['level'] * 100
        epfy['level'] = epfy['level'] * 100

    # rename plev coordinate if required
    if 'plev' in epfy.dims:
        epfy = epfy.rename({'plev': 'level'})
        ua = ua.rename({'plev': 'level'})

    # match pressure levels to smaller dataset
    if len(epfy.level) > len(ua.level):
        epfy = epfy.sel( level = ua.level.values )
    else:
        ua = ua.sel( level = epfy.level.values )

    # create dataset and slice to remove spin-up
    ds = xr.Dataset( {'ubar': ua.ua.mean('lon'), 'epfy': epfy.epfy})
    ds['level'] = ds['level'] / 100
    ds = ds.interp(lat=np.arange(-90,93,3))
    ds = ds.sel(time=slice('2000-06', '2001-05'))

    # subset epfy to match SRIP datasets
    ds = ds.where(ds.level < 1000.)
    ds = ds.where(ds.level > 1.)
    ds = ds.where(ds.lat > -90.)
    ds = ds.where(ds.lat < 90)

    # Calculate DivF
    ds = ef.calculate_divFphi(ds)

    ## SAVE DATASET
    ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/processed_monthly/1.6_pdSST-futArcSIC/{model}_ua_epfy_divF_r{len(ds.ens_ax)}_3x3_futArc.nc')


###################################################################################################

if __name__ == '__main__':

    print('Starting program.')

    # standard models
    # models = ['CESM2', 'CanESM5', 'CNRM-CM6-1', 'EC-EARTH3', 'FGOALS-f3-L', 'MIROC6', 'NorESM2-LM',
    #               'HadGEM3-GC31-MM']
    models = ['HadGEM3-GC31-LL']

    # # delete duplicate models
    # for item in models:
    #     path = f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/processed_monthly/1.6_pdSST-futArcSIC/{item}*.nc'
    #     os.system(f'rm -rf {path}')

    # sort individually
    individual = ['IPSL-CM6A-LR', 'OpenIFS-159', 'OpenIFS-511', 'CESM1-WACCM-SC']
    # not consistent across experiments
    NA = ['E3SMv1', 'ECHAM6.3_AWI']

    for item in models:

        print(f'Processing {item}...')
        process_pamip_monthly(model=item)
        print(f'Model {item} complete.')

    print('Program completed.')
