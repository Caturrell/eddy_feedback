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
from scipy.stats import pearsonr
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
    Input: Xarray Dataset with various dimension labels.
        - Uses aos.FindCoordNames to get coordinate names.
        - Searches for and renames: lon, lat, pres, time.

    Output: Xarray Dataset with renamed dimensions (if needed).
    """

    # Search for dimension labels
    dims = aos.FindCoordNames(ds)

    # Rename time or ensemble dimensions if necessary
    if 't' in ds.dims:
        ds = ds.rename({'t': 'time'})
    elif 'record' in ds.dims:
        ds = ds.rename({'record': 'ens_ax'})

    # Construct rename dictionary based on available keys
    rename_dict = {}
    if 'lon' in dims and dims['lon'] in ds.dims:
        rename_dict[dims['lon']] = 'lon'
    if 'lat' in dims and dims['lat'] in ds.dims:
        rename_dict[dims['lat']] = 'lat'
    if 'pres' in dims and dims['pres'] in ds.dims:
        rename_dict[dims['pres']] = 'level'

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
    Check coordinates are in correct orientation.

    Input: Xarray Dataset
        - Expected standardized dimensions: (time, lon, lat, level)
        - Expected variables: ucomp, vcomp, temp

    Output: Xarray Dataset with adjusted coordinates and variable names.
        - Standardized dimension orientation.
        - Variables renamed to: u, v, t
    """

    # --- Handle pressure levels ---
    if 'level' in ds.dims:
        level = ds['level']
        # Scale down pressure levels if they appear in Pa
        if level.max() > 1000.00:
            ds['level'] = level / 100

        # Ensure levels are ordered from high to low (e.g., [1000 → 0])
        if (ds.level[0] - ds.level[1]) < 0:
            ds = ds.sel(level=slice(None, None, -1))

    # --- Handle latitude orientation ---
    if 'lat' in ds.dims:
        lat = ds['lat']
        # Ensure latitude is ordered from south to north (e.g., [-90 → 90])
        if (lat[0] + lat[1]) > 0:
            ds = ds.sel(lat=slice(None, None, -1))

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

def longitude_adjustment(ds):
    
    """
    Adjust the longitude coordinates of an Xarray dataset to ensure they are in the range [-180, 180].
    """
    
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
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
def seasonal_mean(ds, cut_ends=True, season=None, take_mean=False):
    """
    Calculate seasonal means for an Xarray dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset or xr.DataArray
        Full-year dataset with time dimension
    cut_ends : bool, optional
        If True (default), removes incomplete seasons missing any of their 3 months
    season : str
        Season to compute mean for. Options:
        'djf', 'jfm', 'fma', 'mam', 'amj', 'mjj', 
        'jja', 'jas', 'aso', 'son', 'ond', 'ndj'
    take_mean : bool, optional
        If True, returns time-averaged mean across all seasons
    
    Returns:
    --------
    xr.Dataset or xr.DataArray
        Seasonal means
    """
    season_definitions = {
        'djf': [12, 1, 2],
        'jfm': [1, 2, 3],
        'fma': [2, 3, 4],
        'mam': [3, 4, 5],
        'amj': [4, 5, 6],
        'mjj': [5, 6, 7],
        'jja': [6, 7, 8],
        'jas': [7, 8, 9],
        'aso': [8, 9, 10],
        'son': [9, 10, 11],
        'ond': [10, 11, 12],
        'ndj': [11, 12, 1],
    }

    if season is None or season.lower() not in season_definitions:
        raise ValueError(
            f'Invalid season: {season}. Choose from {list(season_definitions.keys())}'
        )

    season = season.lower()
    months = season_definitions[season]

    # Check required dimensions
    required_dims = ['time', 'lat']
    if 'level' in ds.dims:
        required_dims.append('level')
    if not all(dim in ds.dims for dim in required_dims):
        ds = check_dimensions(ds)

    # Select only the months belonging to this season
    seasonal = ds.sel(time=ds.time.dt.month.isin(months))

    # Assign a season_year coordinate for grouping.
    # For seasons that cross the calendar year boundary, December (and November
    # for NDJ) are attributed to the *following* year so each season lands in
    # one group.
    year_coord = seasonal.time.dt.year.values.copy()
    month_coord = seasonal.time.dt.month.values

    if season == 'djf':
        year_coord[month_coord == 12] += 1
    elif season == 'ndj':
        year_coord[(month_coord == 11) | (month_coord == 12)] += 1

    seasonal = seasonal.assign_coords(season_year=('time', year_coord))

    # Drop any season_year that doesn't contain all 3 months
    if cut_ends:
        print('Cutting incomplete seasons from dataset...')
        unique_years, counts = np.unique(year_coord, return_counts=True)
        complete_years = unique_years[counts == 3]
        seasonal = seasonal.sel(
            time=np.isin(seasonal.season_year.values, complete_years)
        )

    # Group by season_year and average
    seasonal = seasonal.groupby('season_year').mean('time')
    seasonal = seasonal.rename({'season_year': 'time'})

    if take_mean:
        return seasonal.mean('time')
    else:
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
    
    
# Function to compute and add Pearson correlation
def add_correlation(ax, x, y, x_loc=0.05, y_loc=0.95):
    corr, p = pearsonr(x, y)
    ax.text(x_loc, y_loc, f"r = {corr:.2f}, p = {p:.3f}", transform=ax.transAxes, 
            fontsize=14, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7))


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
