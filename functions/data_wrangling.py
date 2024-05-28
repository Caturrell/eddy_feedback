"""
    Python file containing various data wrangling functions.
"""
# pylint: disable=line-too-long
# pylint: disable=wrong-import-position
import sys
import warnings
import numpy as np
import xarray as xr
import xesmf as xe

sys.path.append('/home/users/cturrell/documents/eddy_feedback')
import functions.aos_functions as aos
import functions.eddy_feedback as ef

#==================================================================================================

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



#==================================================================================================

#----------------------
# DATASET SUBSETTING
#----------------------


# Subset to seasonal datasets
def seasonal_dataset(ds, season='djf', save_ds=False, save_location='./ds.nc'):
    """ 
    
    MUST NOT USE WINTER SUBSET FOR MULTI-YEAR DATASETS
        - PAMIP/model data is OK
        - Don't use with reanalysis
    
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
        ds = ds.sel(time=ds.time.dt.month.isin([9, 10, 11]))

    # PAMIP SH summer
    if season == 'jas':
        ds = ds.sel(time=ds.time.dt.month.isin([7, 8, 9]))

    # save new dataset if wanted
    if save_ds:
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

    # If required, check dimensions and variables are labelled correctly
    correct_dims = all(dim_name in ds.dims for dim_name in ['time', 'level', 'lat'])
    if not correct_dims:
        ds = check_dimensions(ds, ignore_dim='lon')

    # remove first Jan and Feb, and final Dec to ensure FULL seasons
    if cut_ends:
        # slice data from first March to final November
        # (assuming dataset is JanYYYY - DecYYYY)
        ds = ds.sel(time=slice(f'{ds.time.dt.year[0].values}-3', \
                                    f'{ds.time.dt.year[-1].values}-11'))

    # resample data to start 1st Dec and take monthly mean
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


#==================================================================================================

#----------------------
# DATASET PROCESSING
#----------------------

# regrid PAMIP data
def regrid_dataset_3x3(ds, check_dims=False):
    """
    Input: Xarray dataset
            - Must contain (lat, lon)
            
    Output: Regridded Xarray dataset 
            to 3 deg lat lon
    
    """
    # Rename
    if check_dims:
        ds = check_dimensions(ds)
    # build regridder
    ds_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90, 93, 3)),
            'lon': (['lon'], np.arange(0,360, 3))
        }
    )

    regridder = xe.Regridder(ds, ds_out, "bilinear")
    ds_new = regridder(ds)

    # verify that the result is the same as regridding each variable one-by-one
    for k in ds.data_vars:
        print(k, ds_new[k].equals(regridder(ds[k])))

    print('Regridding and checks complete. Dataset ready.')
    return ds_new



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
    epfy = xr.open_mfdataset(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/epfy/{model}/*.nc',
                            combine='nested', concat_dim='ens_ax', parallel=True)
    ua = xr.open_mfdataset(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/monthly/pdSST-pdSIC/ua/{model}/*.nc',
                            combine='nested', concat_dim='ens_ax', parallel=True)

    ## Model specific needs:
    if model == 'EC-EARTH3':
        epfy = epfy.mean('lon')

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
    ds.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/processed_monthly/{model}_epfy_ua_r{len(ds.ens_ax)}_3x3.nc')


###################################################################################################

if __name__ == '__main__':

    print('Starting program.')

    models = ['CESM2', 'CanESM5', 'EC-EARTH3', 'FGOALS-f3-L', 'MIROC6', 'NorESM2-LM']

    for item in models:

        print(f'Processing {item}...')
        process_pamip_monthly(model=item)
        print(f'Model {item} complete.')

    print('Program completed.')
