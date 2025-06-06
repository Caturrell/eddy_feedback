import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof
import xcdat
import pdb

import functions.data_wrangling as dw

#---------------------------------------

def eof_calc_alt(data,lats):

    coslat = np.cos(np.deg2rad(lats.values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[np.newaxis, np.newaxis, :]

    solver = Eof(data, weights=wgts, center=True)

    eofs = solver.eofsAsCovariance(neofs=1)
    pc1 = solver.pcs(npcs=1) #, pcscaling=1)  # <---- removed because pcscaling=1 is unit variance : default is un-scaled

    variance_fractions = solver.varianceFraction(neigs=3)

    return eofs, pc1, variance_fractions, solver


def compute_eofs(ds, data_var='u', data_freq='day', apply_vertical_weighting=False, time_frame='all-time', 
                 time_period='simpson', hemisphere='sh', lat_slice=None, calc_mon_avg=False, mask_non_seasonal=False):
    """
    Compute EOFs, principal components (PCs), and variance fractions
    using an alternative EOF function. This function supports different
    anomaly frequencies, time frames, and time period selections.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the variable 'u'. The dataset must have a
        'temporal.departures' method to compute anomalies.
    freq : str, optional
        Frequency for computing anomalies. For the 'all-time' record this may
        be 'day' or 'month'.
        Default is 'day'.
    apply_vertical_weighting : bool, optional
        If True, applies vertical (level-based) weighting to the anomalies over the
        selected pressure levels.
    time_frame : str, optional
        Time frame selection:
          - 'all-time': use all available times without seasonal filtering,
          - 'JJA': use seasonal anomalies by selecting a representative month (e.g. July),
          - 'DJF': use seasonal anomalies by selecting a representative month (e.g. January).
        Default is 'all-time'.
    time_period : str, optional
        Time period for the analysis:
          - 'full': use the entire dataset,
          - '1979-2010': filter the data to the period between January 1, 1979 and December 31, 2010.
        Default is 'full'.

    Returns
    -------
    eofs_ds : xarray.Dataset
        A dataset containing the computed EOFs, PCs, and variance fractions.
    """
    
    #-------------------------------------------------------------------
    # PREAMBLE
    #-------------------------------------------------------------------
    
    # run through my data checker
    # ds = dw.data_checker1000(ds, check_vars=False)
    
    # Filter the dataset if a specific time period is requested.
    if time_period.lower() == 'simpson':
        ds = ds.sel(time=slice('1978-03', '2011-02'))
    else:
        ds = ds
    
    if time_frame.lower() == 'all-time':
        freq_str = 'all-time'
    else:
        freq_str = f'{time_frame.upper()}'
        
    # specify data frequency and the anomaly frequency to be calculated
    if data_freq.lower() == 'day':
        freq_str += '_daily_anom'
    elif data_freq.lower() == 'month':
        freq_str += '_mon_anom'
    elif data_freq.lower() == 'season':
        freq_str += '_szn_anom'
    else:
        raise ValueError("data_freq must be 'day', 'month', or 'season'.")
        
    # Determine file prefix based on the time period.
    if time_period.lower() == 'simpson':
        file_prefix = f'jra55_1979-2010_'
    else:
        file_prefix = 'jra55_'
        
    if hemisphere.lower() == 'nh':
        file_prefix += 'nh_'
        latitude_range = slice(85,20)
    else:
        file_prefix += 'sh_'
        latitude_range = slice(-20,-85)
        
    if calc_mon_avg:
        print('Calculating monthly averages...')
        ds_mon = ds.resample(time='1ME').mean(dim='time')
        ds_mon['time'].encoding = ds['time'].encoding
        ds_mon = ds_mon.bounds.add_missing_bounds(axes='T')
        
        freq_str += '_mon-avg'
        ds = ds_mon
        
    if mask_non_seasonal:
        freq_str += '_Masked'
    
    # Construct the output file name based on whether vertical weighting is applied.
    if apply_vertical_weighting:
        data_file_name = f'{file_prefix}eofs_va_{freq_str}_lat20-85_100-1000hPa.nc'
    else:
        data_file_name = f'{file_prefix}eofs_{freq_str}_lat20-85_100-1000hPa.nc'
    
    save_data_path = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/data/eofs/jra55'
    data_file = os.path.join(save_data_path, data_file_name)
    
    # Load from file if it exists, otherwise compute the EOFs.
    if os.path.isfile(data_file):
        print(f'Loading existing EOFs from {data_file}')
        return xr.open_dataset(data_file)
    
    #-------------------------------------------------------------------
    # CALCULATE ANOMALIES
    #-------------------------------------------------------------------
    
    print(f'Computing EOFs and saving to {data_file}')    
    # Compute departures, weight them, and take the zonal mean.
    var_anoms = ds.temporal.departures(data_var=data_var, freq=data_freq, weighted=True)['u'].mean('lon')
    
    # Select the desired latitudes and pressure levels.
    var_anoms_hem = var_anoms.sel(lat=latitude_range).sel(level=slice(100, 1000)).compute()
    
    # Apply vertical weighting if requested.
    if apply_vertical_weighting:
        dp = var_anoms_hem.level.diff('level')
        var_anoms_hem = (var_anoms_hem * dp).sum('level') / dp.sum('level')
        var_anoms_hem.name = 'u'
        
    # For seasonal cases, either filter or mask by a representative season.
    if time_frame.lower() != 'all-time':
        # Define the months for each season
        season_months = {'jja': [6, 7, 8],
                         'jas': [7, 8, 9], 
                         'djf': [12, 1, 2], 
                         'mam': [3, 4, 5], 
                         'son': [9, 10, 11]}
        key = time_frame.lower()
        if key in season_months:
            season_values = season_months[key]
            
            if mask_non_seasonal:
                # Mask out all non-seasonal months by setting them to NaN
                print('Masking non-seasonal months...')
                pdb.set_trace()
                mask = var_anoms_hem.time.dt.month.isin(season_values)
                var_anoms_hem = var_anoms_hem.where(mask)
            else:
                # Subset to only include seasonal months
                var_anoms_hem = var_anoms_hem.sel(time=var_anoms_hem.time.dt.month.isin(season_values))
        else:  
            raise ValueError("time_frame must be 'all-time' or one of: 'djf', 'mam', 'jja', 'son'")

    #-------------------------------------------------------------------
    # CALCULATE EOFs
    #-------------------------------------------------------------------
    
    # Compute EOFs
    eofs, pc1, variance_fractions, solver = eof_calc_alt(var_anoms_hem, var_anoms_hem.lat)
    
    # Create an xarray Dataset and save the computed outputs.
    eofs_ds = xr.Dataset(coords=eofs.coords)
    eofs_ds['eofs'] = eofs
    eofs_ds['pc1'] = pc1
    eofs_ds['variance_fractions'] = variance_fractions
    
    eofs_ds.to_netcdf(data_file)
    eofs_ds.close()
    return xr.open_dataset(data_file)

def run_eof_calculations_and_plot(ds, data_var='u', data_freq='day', time_frame='all-time', time_period='simpson',
                                  hemisphere='sh', calc_mon_avg=False, mask_non_seasonal=False):
    """
    Compute and plot both the standard and vertically weighted EOFs for a given anomaly frequency,
    time frame, and time period.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the variable 'u'. For seasonal anomalies, this should be
        the full dataset that can later be subset.
    freq : str, optional
        Anomaly frequency for 'all-time' records: 'day' or 'month'. For seasonal time frames,
        this will be overridden to 'season'.
        Default is 'day'.
    time_frame : str, optional
        Time frame: 'all-time', 'JJA', or 'DJF'. Default is 'all-time'.
    time_period : str, optional
        Time period selection: 'full' (all available data) or '1979-2010' to limit the dataset.
        Default is 'full'.
    """
    # Compute both the standard and vertically weighted EOF datasets.
    eofs_ds_standard = compute_eofs(ds, data_var=data_var, data_freq=data_freq, apply_vertical_weighting=False, 
                                    time_frame=time_frame, time_period=time_period, hemisphere=hemisphere,
                                    calc_mon_avg=calc_mon_avg, mask_non_seasonal=mask_non_seasonal)
    eofs_ds_weighted = compute_eofs(ds, data_var=data_var, data_freq=data_freq, apply_vertical_weighting=True, 
                                    time_frame=time_frame, time_period=time_period, hemisphere=hemisphere,
                                    calc_mon_avg=calc_mon_avg, mask_non_seasonal=mask_non_seasonal)
    
    # # For seasonal cases, filter by a representative month.
    # if time_frame.lower() != 'all-time':
    #     # Define the month for each season: JJA uses July and DJF uses January.
    #     season_mid_month = {'jja': 7, 'djf': 1, 'mam': 4, 'son': 10}
    #     key = time_frame.lower()
    #     if key in season_mid_month:
    #         month_value = season_mid_month[key]
    #         eofs_ds_standard = eofs_ds_standard.sel(time=eofs_ds_standard.time.dt.month.isin([month_value]))
    #         eofs_ds_weighted = eofs_ds_weighted.sel(time=eofs_ds_weighted.time.dt.month.isin([month_value]))
    #     else:  
    #         raise ValueError("time_frame must be 'all-time' or one of: 'djf', 'mam', 'jja', 'son'")
    
    # Create side-by-side plots.
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 5))
    
    eofs_standard = eofs_ds_standard['eofs']
    eofs_standard.sel(mode=0).plot.contourf(yincrease=False, levels=21, ax=ax1)
    ax1.set_title(f"EOF 1: {eofs_ds_standard['variance_fractions'][0].values * 100:.1f}%")
    
    eofs_weighted = eofs_ds_weighted['eofs']
    eofs_weighted.sel(mode=0).plot.line(ax=ax2)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_title(f"EOF 1: {eofs_ds_weighted['variance_fractions'][0].values * 100:.1f}%")
    
    # Build a descriptive title based on the frequency, time frame, and time period.
    if time_frame.lower() == 'all-time':
        title_label = 'All time'
    else:
        title_label = time_frame.upper()
    
    if time_period.lower() == 'simpson':
        title_label += " 1979-2010"
    
    if data_freq.lower() == 'day':
        freq_str = 'Daily'
    elif data_freq.lower() == 'month':
        freq_str = 'Monthly'
    elif data_freq.lower() == 'season':
        freq_str = 'Seasonal'
    else:
        raise ValueError("data_freq must be 'day', 'month', or 'season'.")
    
    if hemisphere.lower() == 'nh':
        title_label += ' (NH)'
    else:
        title_label += ' (SH)'
    
    fig.suptitle(f"EOF 1: {title_label} - {freq_str} anomalies", fontsize=16)
    plt.show()

if __name__ == "__main__":
    
    import annular_modes as am
    
    path = '/home/links/ct715/data_storage/reanalysis/jra55_daily'
    data_file = os.path.join(path, 'jra55_uvtw.nc')
    ds = xr.open_mfdataset(data_file, chunks={'time': 30})
    ds = ds.bounds.add_missing_bounds(axes='T')
    
    am.run_eof_calculations_and_plot(ds, data_freq='day', time_frame='jja', calc_mon_avg=True,
                                     time_period='simpson', mask_non_seasonal=True)