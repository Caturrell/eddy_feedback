"""Python function to calculate EP fluxes using aostools package (Martin Jucker). """

# pylint: disable=line-too-long

import glob
import sys
import xarray as xr

sys.path.append('/home/users/cturrell/documents/eddy_feedback')
# pylint: disable=wrong-import-position
import functions.eddy_feedback as ef

#==================================================================================================

#----------------------
# CALCULATION FUNCTIONS
#----------------------

def calculate_pamip_epfluxes(temp, ucomp, vcomp):
    """Function to calculate EP fluxes of PAMIP data.

    Args:
        temp (Xarray DataArray): Temperature
        ucompa (Xarray DataArray): Zonal wind
        vcomp (Xarray DataArray): Meridional wind

    Returns:
        Xarray Dataset: Original Dataset containing EP fluxes
    """

    # Combine datasets into one
    ds = xr.Dataset(data_vars={'ta': temp.ta, 'ua': ucomp.ua, 'va': vcomp.va})

    # Calculating EP fluxes
    ds = ef.calculate_epfluxes_ubar(ds, pamip_data=True)

    return ds


#==================================================================================================

if __name__ == '__main__':

    # Set time of day for program echo
    from datetime import datetime
    now = datetime.now().strftime("%H:%M:%S")
    print("Current Time =", now)

    print('Importing CanESM5 datasets...')
    ta_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/ta/*')
    ua_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/ua/*')
    va_files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/va/*')

    print(f'ta:{len(ta_files)}, ua:{len(ua_files)}, va:{len(va_files)}')

    for i, file in enumerate(ta_files):
        # reset time for counter
        now = datetime.now().strftime("%H:%M:%S")

        # import each var as dataset
        ta = xr.open_mfdataset(file)
        ua = xr.open_mfdataset(ua_files[i])
        va = xr.open_mfdataset(va_files[i])

        # Start function
        print(f'[{now}]: Datasets ({i+1}) loaded. Calculations starting...')
        dataset = calculate_pamip_epfluxes(ta, ua, va)
        print('Calculations complete. Now saving dataset...')

        # Save new dataset
        dataset.to_netcdf(f'/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/all/CanESM5_uvt_epfluxes_3x3_ens{i+1}.nc')
        print(f'[{now}]: CanESM5 dataset saved. {i+1} iteration completed.')

    print(f'[{now}]: Loop completed. Now creating singular dataset with EFP variables.')

    # subset dataset and save it
    files = glob.glob('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/CanESM5_3x3/all/*.nc')
    dataset = xr.open_mfdataset(files, combine='nested', concat_dim='ens_ax', parallel=True)
    
    print('Dataset Loaded. Saving dataset...')
    dataset = dataset[['ubar', 'div1']]
    dataset['level'] = dataset['level'] / 100
    dataset.to_netcdf('/gws/nopw/j04/arctic_connect/cturrell/PAMIP_data/regridded/3x3_ef_Can.nc')

    print(f'[{now}]: PROGRAM COMPLETED.')
