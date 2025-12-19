import numpy as np
from eofs.standard import Eof
from tqdm import tqdm
from scipy import signal
from scipy import optimize as scipyo
import scipy
import numpy as np
import statsmodels.tsa.stattools as sm 
import scipy.signal as signal
import xarray as xar
import os
import SIT_aostools.climate as aoscli
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import SIT_eddy_plotting_functions as epf
import shutil

import pdb

def compute_segmented_power_spectrum(data, scaling, power_spec_ds, dim='time', segment_length=256, overlap=128):
    """
    Compute the power spectrum using overlapping segments and a Hanning window.

    Parameters:
        data (xarray.DataArray): Input time series.
        dim (str): The time dimension (default: 'time').
        segment_length (int): Length of each segment in time units (e.g., days).
        overlap (int): Overlapping length between segments (e.g., days).

    Returns:
        freqs (xarray.DataArray): Frequency values.
        power_spectrum (xarray.DataArray): Averaged power spectrum.
    """
    time_values = data[dim].values
    dt = np.median(np.diff(time_values))/np.timedelta64(1, 's')  # Time step
    
    if np.isclose(dt, 86400.):
        dt = 1.
    else:
        raise NotImplemented('needs to be daily data')    

    fs = 1 / dt  # Sampling frequency in cycles per day

    # Convert segment length from days to number of points
    segment_size = int(segment_length / dt)
    overlap_size = int(overlap / dt)

    # Compute power spectrum using Welch’s method
    freqs, power_spectrum = signal.welch(
        data.values, fs=fs, 
        window='hann',  # Hanning window
        nperseg=segment_size, 
        noverlap=overlap_size,
        scaling=scaling
    )

    freqs2, t, complex_spectrum = signal.stft(data.values, fs=fs, window='hann', nperseg=segment_size, noverlap=overlap_size, scaling='psd') 

    complex_spectrum_amp= np.abs(complex_spectrum)**2.

    dim_names = [key for key in power_spec_ds.coords.keys()]

    frequency_name = f'frequency_{dim}'

    if frequency_name not in dim_names:
        power_spec_ds.coords[frequency_name] = ((frequency_name), freqs)


    if dim not in dim_names:
        power_spec_ds.coords[dim] = ((dim), t)


    power_spec_ds[f'{data.name}_power_spec_welch'] = ((frequency_name), power_spectrum)

    power_spec_ds[f'{data.name}_power_spec_stft'] = ((frequency_name),np.mean(complex_spectrum_amp,axis=1))

    power_spec_ds[f'{data.name}_fourier_coeffs_stft'] = ((frequency_name, dim),complex_spectrum)

    return power_spec_ds

def compute_cospectrum(data1, data2, power_spec_ds, dim='time', segment_length=256, overlap=128, ucomp_name='ucomp_s_all_time', div1_name=''):
    """
    Compute the cospectrum between two time series using Welch's method.

    Parameters:
        data1 (xarray.DataArray): First input time series.
        data2 (xarray.DataArray): Second input time series.
        dim (str): The time dimension (default: 'time').
        segment_length (int): Length of each segment in time units (e.g., days).
        overlap (int): Overlapping length between segments (e.g., days).

    Returns:
        freqs (xarray.DataArray): Frequency values.
        cospectrum (xarray.DataArray): Cospectral density.
    """
    if not np.array_equal(data1[dim], data2[dim]):
        raise ValueError("The two time series must have the same time coordinate.")

    time_values = data1[dim].values
    dt = np.median(np.diff(time_values))/np.timedelta64(1, 's')  # Time step
    
    if np.isclose(dt, 86400.):
        dt = 1.
    else:
        raise NotImplemented('needs to be daily data')    

    fs = 1 / dt  # Sampling frequency in cycles per day

    # Convert segment length from days to number of points
    segment_size = int(segment_length / dt)
    overlap_size = int(overlap / dt)

    # # Compute cross-power spectral density (CSD)
    # freqs, csd = signal.csd(
    #     data1.values, data2.values, fs=fs, 
    #     window='hann',  # Hanning window
    #     nperseg=segment_size, 
    #     noverlap=overlap_size,
    #     scaling='density'
    # )

    # power_spectrum_ucomp_da = power_spec_ds[f'{ucomp_name}_power_spec_welch'].values

    # csd = csd / power_spectrum_ucomp_da

    # # Extract the cospectrum (real part of the CSD)
    # cospectrum = np.real(csd)
    # cospectrum_im = np.imag(csd)

    csd_stft_unnorm = np.mean(np.conjugate(power_spec_ds[f'{ucomp_name}_fourier_coeffs_stft']) * power_spec_ds[f'{div1_name}_fourier_coeffs_stft'], axis=1)

    csd_stft = csd_stft_unnorm/power_spec_ds[f'{ucomp_name}_power_spec_stft']

    freqs_co, coher = signal.coherence(data1.values, data2.values, fs=fs, window='hann', nperseg=segment_size, noverlap=overlap_size)

    coher_stft = np.abs(csd_stft_unnorm)**2. / (power_spec_ds[f'{ucomp_name}_power_spec_stft'] * power_spec_ds[f'{div1_name}_power_spec_stft'])

    # Convert to xarray DataArray
    # power_spec_ds[f'{data1.name}_{data2.name}_cospec_welch'] = (('frequency'), csd)
    # power_spec_ds[f'{data1.name}_{data2.name}_coher_welch'] = (('frequency'), coher)

    power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft'] = (csd_stft.dims, csd_stft.values)
    power_spec_ds[f'{data1.name}_{data2.name}_coher_stft'] = (coher_stft.dims, coher_stft.values)


    return power_spec_ds

def compute_phase_difference(power_spec_ds, div1_name, ucomp_name, time_name):

    div1_ucomp_phase_diff = np.mean(np.angle(power_spec_ds[f'{div1_name}_fourier_coeffs_stft']/power_spec_ds[f'{ucomp_name}_fourier_coeffs_stft'], deg=True),axis=1) #Have to take the phase difference in every time period using angle, and then average over the time periods.

    frequency_name = f'frequency_{time_name}'

    omega_freqs = 2.*np.pi*power_spec_ds[frequency_name]

    timescale_from_phase_diff, _ = scipyo.curve_fit(arctan_omega,omega_freqs.values,np.deg2rad(div1_ucomp_phase_diff))

    timescale_from_phase_diff_days = timescale_from_phase_diff[0] #frequency in per day so timescale in days

    where_low_freq = np.where(power_spec_ds[frequency_name].values<=0.025)[0] #L+H only perform fit below 0.025 day^-1. This isn't specified whether that's angular frequency or f, but if it's angular frequency that means we fit based on only TWO data points, which is a bit rubbish. So assume it's f and we get seven points. That's still not great, but OK.

    logging.info(f'Using only {np.shape(where_low_freq)[0]} points for fit of tau')

    omega_freqs_lim = omega_freqs[where_low_freq]
    div1_ucomp_phase_diff_lim = div1_ucomp_phase_diff[where_low_freq]   

    timescale_from_phase_diff2, _ = scipyo.curve_fit(arctan_omega,omega_freqs_lim.values,np.deg2rad(div1_ucomp_phase_diff_lim))

    timescale_from_phase_diff_days2 = timescale_from_phase_diff2[0] #frequency in per day so timescale in days

    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff'] = ((frequency_name), div1_ucomp_phase_diff)

    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_1'] = timescale_from_phase_diff_days

    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_2'] = timescale_from_phase_diff_days2

    real_cospec_stft = np.real(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft'])
    imag_cospec_stft = np.imag(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft'])

    lin_reg_real = scipy.stats.linregress(omega_freqs[where_low_freq], real_cospec_stft[where_low_freq])

    lin_reg_imag = scipy.stats.linregress(omega_freqs[where_low_freq], imag_cospec_stft[where_low_freq])    

    tau_estimate_3 = lin_reg_imag.slope / lin_reg_real.intercept

    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'] = tau_estimate_3


    return power_spec_ds


def eof_calc_alt(data,lats):

    coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
    wgts = np.sqrt(coslat)[np.newaxis, np.newaxis, :]

    solver = Eof(data, weights=wgts, center=True)

    eofs = solver.eofsAsCovariance(neofs=3)
    pc1 = solver.pcs(npcs=3, pcscaling=1)

    variance_fractions = solver.varianceFraction(neigs=3)

    return eofs, pc1, variance_fractions, solver

def cross_correlation(series1, series2, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    len_series1 = series1.shape[0]
    len_series2 = series2.shape[0]
    assert(len_series1==len_series2)

    series1_start_idx = max_lag
    series1_end_idx   = len_series1-max_lag

    # for lag in lags:
    #     series2_start_idx = max_lag+lag
    #     series2_end_idx   = len_series2-max_lag+lag
    #     print(lag, 1, series1_start_idx,series1_end_idx, lag)
    #     print(lag, 2, series2_start_idx,series2_end_idx, lag)    

    # corr = [np.corrcoef(series1[series1_start_idx:series1_end_idx], series2[max_lag+lag:len_series2-max_lag+lag])[0, 1] for lag in lags]

    corr = []

    for lag in lags:

        series_1_short = series1[series1_start_idx:series1_end_idx]
        series_2_short = series2[max_lag+lag:len_series2-max_lag+lag]

        valid_idx = np.isfinite(series_1_short) & np.isfinite(series_2_short)


        corr.append(np.corrcoef(series_1_short[valid_idx], series_2_short[valid_idx])[0, 1])

    return corr, lags

def sm_cross_correlation(series1, series2, max_lag):
    ccf_values = sm.ccf(series1, series2, nlags=max_lag)
    return ccf_values

def ep_flux_calc(dataset, output_file, force_ep_flux_recalculate, include_udt_rdamp, omega_rot_rate, a0):

    if os.path.isfile(output_file) and not force_ep_flux_recalculate:
        logging.info('attempting to read in data')
        epflux_ds = xar.open_mfdataset(output_file, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated data - CALCULATING')

        dataset_vars = [key for key in dataset.variables.keys()]

        omega_present = 'omega' in dataset_vars

        epflux_ds = xar.Dataset(coords=dataset.coords)

        # logging.info('calculating ep flux')
        # if omega_present:
        #     logging.info('calculating FULL ep flux')

        #     epflux_ds['ep1'], epflux_ds['ep2'], epflux_ds['div1'], epflux_ds['div2'], epflux_ds['dthdp_bar'] = aoscli.ComputeEPfluxDivXr(dataset['ucomp'], dataset['vcomp'], dataset['temp'], do_ubar=True, w = dataset['omega']/100., ref='instant') #Omega is divided by 100 as code expects it in units of hPa/s, but Isca gives out in Pa/s.
        #     logging.info('now calculating EP flux from waves 1-3 only')
        #     epflux_ds['ep1_123'], epflux_ds['ep2_123'], epflux_ds['div1_123'], epflux_ds['div2_123'], epflux_ds['dthdp_bar_123'] = aoscli.ComputeEPfluxDivXr(dataset['ucomp'], dataset['vcomp'], dataset['temp'], do_ubar=True, w = dataset['omega']/100., ref='instant', wave=[1,2,3]) #Omega is divided by 100 as code expects it in units of hPa/s, but Isca gives out in Pa/s.            
        #     for var_name_to_decompose in ['ep1', 'ep2', 'div1', 'div2']:
        #         epflux_ds[f'{var_name_to_decompose}_gt3'] = epflux_ds[f'{var_name_to_decompose}'] - epflux_ds[f'{var_name_to_decompose}_123']
        # else:
        #     logging.info('CANNOT calculate full ep flux as omega not present')


        logging.info('Calculating QG ep flux...')
        epflux_ds['ep1_QG'], epflux_ds['ep2_QG'], epflux_ds['div1_QG'], epflux_ds['div2_QG'], epflux_ds['dthdp_bar_QG'] = aoscli.ComputeEPfluxDivXr(dataset['ucomp'], dataset['vcomp'], dataset['temp'], do_ubar=False, ref='instant') #Omega is not present and do_ubar = False, and this gives us QG.
        logging.info('EP fluxes calculated. Now calculating QG EP fluxes from waves 1-3 only...')        
        epflux_ds['ep1_QG_123'], epflux_ds['ep2_QG_123'], epflux_ds['div1_QG_123'], epflux_ds['div2_QG_123'], epflux_ds['dthdp_bar_QG_123'] = aoscli.ComputeEPfluxDivXr(dataset['ucomp'], dataset['vcomp'], dataset['temp'], do_ubar=False, ref='instant', wave=[1,2,3]) #Omega is not present and do_ubar = False, and this gives us QG.

        for var_name_to_decompose in ['ep1_QG', 'ep2_QG', 'div1_QG', 'div2_QG']:
            epflux_ds[f'{var_name_to_decompose}_gt3'] = epflux_ds[f'{var_name_to_decompose}'] - epflux_ds[f'{var_name_to_decompose}_123']
        logging.info('COMPLETED: QG EP fluxes from waves 1-3 only.')        


        if not omega_present:
            epflux_ds['dthdp_bar'] = epflux_ds['dthdp_bar_QG']

        epflux_ds['inv_dthdp_bar'] = 1./epflux_ds['dthdp_bar']

        logging.info('preparing data for vbarstar calc')
        # vstar_in_dict = {'temp':dataset['temp'].values,
        #                 'vcomp':dataset['vcomp'].values,
        #                 'pfull':dataset['pfull'].values,}

        logging.info('calculating vbarstar')
        # epflux_ds['vbarstar'] = (('time', 'pfull', 'lat'), aoscli.ComputeVstar(vstar_in_dict,wave=0)) #the wave=0 here is to stop computeVertEddy from doing the wavenumber decomposition
        epflux_ds['vbarstar'] = aoscli.ComputeVstar(dataset, ref='instant')

        epflux_ds['fvbarstar'] = epflux_ds['vbarstar'] * 2.*omega_rot_rate * np.sin(np.deg2rad(dataset['lat']))

        if omega_present:
            logging.info('calculating omegabarstar')
            epflux_ds['omegabarstar'] = aoscli.ComputeWstarXr(dataset['omega']/100., dataset['temp'], dataset['vcomp'], pres='pfull', ref='instant') #As previously, omega should be in hPa/s, hence dividing by 100

        logging.info('calculating other terms')
        dphi = np.gradient(np.deg2rad(dataset['lat']).values)[np.newaxis,np.newaxis,:]

        dudphi = aoscli.FlexiGradPhi(dataset['ucomp'].mean('lon')*np.cos(np.deg2rad(dataset['lat'])).values, dphi)

        epflux_ds['dudphi'] = (('time', 'pfull', 'lat'), dudphi)

        epflux_ds['1oacosphi_dudphi'] = epflux_ds['dudphi'] * (1./(a0*np.cos(np.deg2rad(dataset['lat']))))

        dp = np.gradient(dataset['pfull'].values*100.)[np.newaxis,:,np.newaxis]


        dudp = aoscli.FlexiGradP(dataset['ucomp'].mean('lon').values,dp)

        epflux_ds['dudp'] = (('time', 'pfull', 'lat'), dudp)

        epflux_ds['vbarstar_1oacosphi_dudphi'] =  (epflux_ds['vbarstar']*epflux_ds['1oacosphi_dudphi'])

        if omega_present:
            epflux_ds['omegabarstar_dudp'] = epflux_ds['omegabarstar']*epflux_ds['dudp']
            epflux_ds['total_tend'] = epflux_ds['fvbarstar'] - epflux_ds['vbarstar_1oacosphi_dudphi'] - epflux_ds['omegabarstar_dudp']  + epflux_ds['div1']/86400. + epflux_ds['div2']/86400. 

        epflux_ds['total_tend_QG'] = epflux_ds['fvbarstar'] + epflux_ds['div1_QG']/86400. + epflux_ds['div2_QG']/86400. 

        if include_udt_rdamp:
            epflux_ds['total_tend'] = epflux_ds['total_tend'] + dataset['udt_rdamp'].mean('lon')
            epflux_ds['total_tend_QG'] = epflux_ds['total_tend_QG'] + dataset['udt_rdamp'].mean('lon')            

        epflux_ds['delta_ubar_dt'] = (dataset['ucomp'][-1,...].mean('lon').squeeze() - dataset['ucomp'][0,...].mean('lon').squeeze())/ (86400.*(dataset['time'][-1].values - dataset['time'][0].values).days)

        logging.info(f'Writing EP flux data etc to file: {output_file}')
        
        
        size_mb = epflux_ds.nbytes / (1024**2)
        logging.info(f'Dataset size: {size_mb:.2f} MB')
        epflux_ds.to_netcdf(output_file)

        # scratch_file = f'/tmp/{os.path.basename(output_file)}'

        # encoding = {var: {
        #     'zlib': True, 
        #     'complevel': 1,  # Lower compression = faster on slow filesystems
        #     'shuffle': True
        # } for var in epflux_ds.data_vars}

        # logging.info(f'Writing to scratch: {scratch_file}')
        # epflux_ds.to_netcdf(scratch_file, engine='netcdf4', encoding=encoding)

        # logging.info(f'Copying from scratch to final location')
        # shutil.move(scratch_file, output_file)
        
        logging.info('FINISHED writing EP flux data etc to file')

        epflux_ds.close()
        epflux_ds = xar.open_mfdataset(output_file, decode_times=True)

    return epflux_ds

def efp_calc(output_efp_file, force_efp_recalculate, dataset, vars_to_correlate, exp_type, season_month_dict, use_500hPa_only=False): 

    if use_500hPa_only:
        pre_path = output_efp_file.split('.nc')[0]
        output_efp_file = pre_path+'_500hPa.nc'

    if os.path.isfile(output_efp_file) and not force_efp_recalculate:
        logging.info('attempting to read in EFP data')
        efp_output_ds = xar.open_mfdataset(output_efp_file, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated EFP data - CALCULATING')

        start_year = dataset.time.dt.year[0].values
        end_year = dataset.time.dt.year[-1].values
        if exp_type!='isca':
            dataset_cut_ends = dataset.sel(time=slice(f'{start_year:04d}-03', f'{end_year:04d}-11'))
        else:
            dataset_cut_ends = dataset
        
        efp_output_ds = xar.Dataset()
        efp_output_ds.coords['lat'] = (('lat'), dataset['lat'].values)
        efp_output_ds.coords['pfull'] = (('pfull'), dataset['pfull'].values) 

        season_list = [season_val for season_val in season_month_dict.keys()]

        all_time_season_list = season_list+['all_time']
        
        for time_frame in tqdm(all_time_season_list):

            if time_frame =='DJF':
                seasonal = dataset_cut_ends.resample(time='QS-DEC').mean('time')
                seasonal = seasonal.sel(time=seasonal.time.dt.month == 12)
            elif time_frame=='NDJ':
                seasonal = dataset_cut_ends.resample(time='QS-NOV').mean('time')
                seasonal = seasonal.sel(time=seasonal.time.dt.month == 11)    
            elif time_frame=='all_time':
                seasonal = dataset_cut_ends.groupby('time.year').mean('time').rename({'year': 'time'})
            else:
                seasonal = dataset.sel(time=dataset.time.dt.month.isin(season_month_dict[time_frame]))
                seasonal = seasonal.groupby('time.year').mean('time').rename({'year': 'time'})

            for hemisphere in ['n', 's']:

                if np.all(dataset.lat.diff('lat').values>0.):
                    if hemisphere=='n':                
                        efp_lat_slice = slice(25., 75.)
                    else:
                        efp_lat_slice = slice(-75., -25.)
                else:
                    if hemisphere=='n':
                        efp_lat_slice = slice(75., 25.)
                    else:
                        efp_lat_slice = slice(-25., -75.)
    
                if np.all(dataset.pfull.diff('pfull').values>0.):
                    if use_500hPa_only:
                        pfull_slice = slice(500.-0.1, 500.+0.1)                        
                    else:
                        pfull_slice = slice(200.,600.)
                else:
                    if use_500hPa_only:
                        pfull_slice = slice(500.+0.1, 500.-0.1)                        
                    else:                        
                        pfull_slice = slice(600.,200.)

                efp_ds = seasonal

                for var2_to_correlate in tqdm(vars_to_correlate+['ucomp']):
                    for var_to_correlate in vars_to_correlate + ['ucomp']:

                        corr_var_name = f'{var_to_correlate}_{var2_to_correlate}_{hemisphere}_{time_frame}_corr'

                        opposite_corr_var_name = f'{var2_to_correlate}_{var_to_correlate}_{hemisphere}_{time_frame}_corr'

                        eof_output_ds_vars = [key for key in efp_output_ds.variables.keys()]

                        if opposite_corr_var_name in eof_output_ds_vars:
                            logging.info(f'skipping efp for {corr_var_name} as already present by a different name')

                            efp_output_ds[corr_var_name] = efp_output_ds[opposite_corr_var_name]

                            efp_output_ds[f'efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}_{time_frame}'] = efp_output_ds[f'efp_{var2_to_correlate}_{var_to_correlate}_{hemisphere}_{time_frame}']            
                            efp_output_ds[f'signed_efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}_{time_frame}'] = efp_output_ds[f'signed_efp_{var2_to_correlate}_{var_to_correlate}_{hemisphere}_{time_frame}']             

                        else:

                            # logging.info(f'calculating efp for {corr_var_name}')

                            data_to_correlate = efp_ds[var_to_correlate]

                            if 'lon' in data_to_correlate.dims:
                                data_to_correlate = data_to_correlate.mean('lon')

                            data2_to_correlate = efp_ds[var2_to_correlate]

                            if 'lon' in data2_to_correlate.dims:
                                data2_to_correlate = data2_to_correlate.mean('lon')

                            # any_nans_in_data1 = np.any(np.isnan(data_to_correlate))
                            # any_nans_in_data2 = np.any(np.isnan(data2_to_correlate))         

                            # if any_nans_in_data1 or any_nans_in_data2:
                            #     valid_mask = np.isfinite(data_to_correlate) & np.isfinite(data2_to_correlate)

                            # std_1 = data_to_correlate.std(dim='time')
                            # std_2 = data2_to_correlate.std(dim='time')

                            # valid_mask = (std_1!=0) & (std_2!=0.)

                            # efp_output_ds[corr_var_name] = xar.corr(data_to_correlate.where(valid_mask), data2_to_correlate.where(valid_mask), dim='time')

                            efp_output_ds[corr_var_name] = xar.corr(data_to_correlate, data2_to_correlate, dim='time')

                            corr_slice = efp_output_ds[corr_var_name].sel(lat=efp_lat_slice, pfull=pfull_slice)

                            take_level_mean = True

                            corr_slice = corr_slice**2 #need to square for EFP average

                            signed_corr_slice = corr_slice*np.abs(corr_slice) #makes same calculation but retains the sign

                            if take_level_mean:
                                corr_av = corr_slice.mean('pfull')
                                signed_corr_av = signed_corr_slice.mean('pfull')
                            else:
                                corr_av = corr_slice
                                signed_corr_av = signed_corr_slice

                            weights = np.cos(np.deg2rad(corr_av.lat))

                            efp = corr_av.weighted(weights).mean('lat')
                            signed_efp = signed_corr_av.weighted(weights).mean('lat')            

                            efp_output_ds[f'efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}_{time_frame}'] = efp.values

                            efp_output_ds[f'signed_efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}_{time_frame}'] = signed_efp.values                            

                            logging.info(f'efp {var_to_correlate} {var2_to_correlate} {time_frame} {hemisphere} = {efp.values}')

        efp_output_ds.to_netcdf(output_efp_file)    

        efp_output_ds.close()
        efp_output_ds = xar.open_mfdataset(output_efp_file, decode_times=True)

    return efp_output_ds

def calculate_anomalies(dataset, var_list, subtract_annual_cycle, output_anom_file, force_anom_recalculate):

    if os.path.isfile(output_anom_file) and not force_anom_recalculate:
        logging.info('attempting to read in anom data')
        anom_ds = xar.open_mfdataset(output_anom_file, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated anom data - CALCULATING')

        anom_file_list = []

        var_anoms1, orig_var1 = calculate_anom_one_var(dataset, subtract_annual_cycle, var_list[0])

        anom_ds = xar.Dataset(coords = var_anoms1.coords)

        anom_ds[f'{var_list[0]}_anom'] = var_anoms1
        anom_ds[f'{var_list[0]}_orig'] = orig_var1    

        single_var_file = output_anom_file.split('nc')[0]+f'_{var_list[0]}.nc'

        logging.info(f'writing {var_list[0]} anoms to file')
        anom_ds.to_netcdf(single_var_file)
        anom_file_list.append(single_var_file)
        anom_ds.close()
        logging.info('completed writing anoms to file')         

    
        for eof_var in var_list[1:]:
            var_anoms, orig_var = calculate_anom_one_var(dataset, subtract_annual_cycle, eof_var)

            anom_ds = xar.Dataset(coords = var_anoms.coords)

            anom_ds[f'{eof_var}_anom'] = var_anoms
            anom_ds[f'{eof_var}_orig'] = orig_var   

            single_var_file = output_anom_file.split('nc')[0]+f'_{eof_var}.nc'
            logging.info(f'writing {eof_var} anoms to file')

            anom_ds.to_netcdf(single_var_file)
            anom_file_list.append(single_var_file)
            anom_ds.close()

            logging.info('completed writing anoms to file')         

        anom_ds = xar.open_mfdataset(anom_file_list, parallel=False)

        logging.info('writing combined anoms to file')
        anom_ds.to_netcdf(output_anom_file)  
        logging.info('completed writing anoms to file')         
        anom_ds.close()
        logging.info('reopening anoms file')                 
        anom_ds = xar.open_mfdataset(output_anom_file, decode_times=True)

    return anom_ds
    

def calculate_anom_one_var(dataset, subtract_annual_cycle, data_var):

    if subtract_annual_cycle:
        logging.info(f'subtracting annual cycle from {data_var}')
        var_anoms = dataset.temporal.departures(data_var = data_var, freq='day', weighted=True)[data_var]
        orig_var = dataset[data_var]
        if var_anoms['time'].shape[0]!=orig_var['time'].shape[0]:
            logging.info('inconsistent times before and after temporal departures. Likely leap day problem.')
            orig_var = orig_var.sel(time=~((orig_var.time.dt.month == 2) & (orig_var.time.dt.day == 29)))

            if var_anoms['time'].shape[0]==orig_var['time'].shape[0]:
                logging.info('problem solved')
            else:
                raise NotImplementedError('Why are number of times different?')
            
        if 'lon' in var_anoms.dims:
            var_anoms = var_anoms.mean('lon')
            orig_var = orig_var.mean('lon')
    else:
        if 'lon' in dataset[data_var].dims:
            var_zm = dataset[data_var].mean('lon')
        else:
            var_zm = dataset[data_var]                
        var_zm_time = var_zm.mean('time')
        var_anoms = var_zm - var_zm_time
        orig_var = var_zm    

    return var_anoms, orig_var

def propagate_missing_data_to_all_vars(anom_ds, eof_vars):
    '''The EOF projectfield method requires that each of the variables
    we project must have missing data in the same place as the field 
    we are projecting onto. This is quite difficult to ensure
    particularly when I have added nans to div1 and div2 when the dtheta/dp
    is vanishingly small. To get around this, this function looks where the 
    nan values are in each of the anomaly fields and makes sure every other 
    variable also has missing data there.'''

    anom_coord_list = [key for key in anom_ds.coords.keys()]
    # anom_var_list = [key for key in anom_ds.variables.keys() if key not in anom_coord_list and '_anom' in key]
    anom_var_list = [f'{key}_anom' for key in eof_vars]

    # nan_location_dict = {}
    n_nans_list = []

    # for anom_var in anom_var_list:
    #     nan_location_dict[anom_var] = np.where(np.isnan(anom_ds[anom_var].values))[0]

    # for anom_var in anom_var_list:
    #     anom_data = anom_ds[anom_var].values
    #     for anom_var_nan_var in anom_var_list:
    #         nan_location_to_apply = nan_location_dict[anom_var_nan_var]
    #         if nan_location_to_apply.shape[0]>0:
    #             pdb.set_trace()
    #             anom_data[nan_location_to_apply] = np.nan
    #     anom_ds[anom_var] = (anom_ds[anom_var].dims, anom_data)
    #     n_nans_list.append(np.where(np.isnan(anom_ds[anom_var]))[0].shape[0])

    for anom_var in anom_var_list:
        for anom_var_nan_var in anom_var_list:
            anom_ds[anom_var] = anom_ds[anom_var].where(np.isfinite(anom_ds[anom_var_nan_var]))
        n_nans_list.append(np.where(np.isnan(anom_ds[anom_var]))[0].shape[0])

    all_anom_vars_same_number_nan = np.all(np.asarray(n_nans_list)==n_nans_list[0])

    if all_anom_vars_same_number_nan:
        logging.info(f'successfully propagated missing data from all vars to all other vars to enable eof projection, so each field now has {n_nans_list[0]} nans')
    else:
        raise ValueError('anom vars contain differing number of nans, which means the projection will not work')

    # for anom_var in anom_var_list:
    #     for anom_var_nan_var in anom_var_list:
    #         all_same = np.all(np.isnan(anom_ds[anom_var].values) == np.isnan(anom_ds[anom_var_nan_var].values))
    #         print(f'{all_same} for {anom_var} and {anom_var_nan_var}')

    return anom_ds


def eof_calc(exp_type, output_eof_file, force_eof_recalculate, dataset, pfull_slice, subtract_annual_cycle, eof_vars, n_eofs, season_month_dict, anom_ds, propagate_all_nans):

    if np.all(dataset.lat.diff('lat')<0.):
        hemisphere_slice_dict = {'n':slice(80.,10.),
                                's':slice(-10., -80.),                          
                                }
    else:
        hemisphere_slice_dict = {'n':slice(10.,80.),
                                's':slice(-80., -10.),                             
                                }        

    if propagate_all_nans:
        output_eof_file_use = output_eof_file.split('.nc')[0]+'_prop_nans.nc'
    else:
        output_eof_file_use = output_eof_file

    if os.path.isfile(output_eof_file_use) and not force_eof_recalculate:
        logging.info('attempting to read in EOF data')
        eof_ds = xar.open_mfdataset(output_eof_file_use, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated EOF data - CALCULATING')

        logging.info('calculating anomalies')
        if exp_type=='held_suarez':
            u_anoms = dataset['ucomp'].mean('lon') - dataset['ucomp'].mean(('time','lon'))
        else:
            # u_anoms_time = dataset.temporal.departures(data_var = 'ucomp', freq='day', weighted=False)['ucomp'].mean('lon')
            u_anoms_time = anom_ds['ucomp_anom']

        season_list = [season_val for season_val in season_month_dict.keys()]

        all_time_season_list = season_list+['all_time']

        eof_ds = xar.Dataset()
        eof_ds.coords['time'] = ('time', u_anoms_time.time.values)
        eof_ds.coords['pfull'] = ('pfull', u_anoms_time.sel(pfull=pfull_slice).pfull.values)  
        eof_ds.coords['lat'] = ('lat', u_anoms_time.sel(lat=hemisphere_slice_dict['n']).lat.values)                

        for season_val in season_list:
            u_anoms_season = u_anoms_time.where(anom_ds['time'].dt.month.isin(season_month_dict[season_val]), drop=True)
            eof_ds.coords[f'time_{season_val}'] = (f'time_{season_val}', u_anoms_season.time.values)

        eof_ds.coords['eof_num'] = ('eof_num', np.arange(n_eofs))

        ucomp_solver_dict = {}
        ucomp_500_solver_dict = {}        
        ucomp_va_solver_dict = {}        
        for season_val in all_time_season_list:
            ucomp_solver_dict[season_val] = {'n':{}, 's':{}}
            ucomp_va_solver_dict[season_val] = {'n':{}, 's':{}}
            ucomp_500_solver_dict[season_val] = {'n':{}, 's':{}}

        logging.info('about to propagate nans to all fields')
        if propagate_all_nans:
            anom_ds = propagate_missing_data_to_all_vars(anom_ds, eof_vars)

        for eof_var in tqdm(eof_vars):

            logging.info('loading anomalies from anom_ds')

            var_anoms = anom_ds[f'{eof_var}_anom'].load()
            orig_var = anom_ds[f'{eof_var}_orig'].load()           

            for hemisphere in ['n','s']:

                for time_frame in all_time_season_list:
                    var_anoms_hem = var_anoms.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)
                    
                    if time_frame!='all_time':
                        var_anoms_hem = var_anoms_hem.where(eof_ds['time'].dt.month.isin(season_month_dict[time_frame]), drop=True) 
                        time_dim_name = f'time_{time_frame}'
                    else:
                        time_dim_name = 'time'

                    orig_var_hem = orig_var.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)
                    
                    if time_frame!='all_time':
                        orig_var_hem = orig_var_hem.where(eof_ds['time'].dt.month.isin(season_month_dict[time_frame]), drop=True)
                        
                    orig_var_hem = orig_var_hem.mean('time')   

                    logging.info(f'calculating EOFs for {eof_var} in {hemisphere} hemisphere in {time_frame}')
                    eofs, pc1, variance_fractions, solver = eof_calc_alt(var_anoms_hem.values,var_anoms_hem.lat.values)

                    var_anoms_hem_500 = var_anoms_hem.sel(pfull=500., method='nearest')

                    logging.info(f'calculating EOFs at 500hPa for {eof_var} in {hemisphere} hemisphere in {time_frame}')
                    eofs_500, pc1_500, variance_fractions_500, solver_500 = eof_calc_alt(var_anoms_hem_500.values,var_anoms_hem_500.lat.values)                    

                    logging.info(f'vertically integrating {eof_var}')
                    va_var_anoms_hem = vert_integrate(var_anoms_hem)

                    logging.info(f'calculating VA EOFs for {eof_var} in {hemisphere} hemisphere in {time_frame}')
                    eofs_va, pc1_va, variance_fractions_va, solver_va = eof_calc_alt(va_var_anoms_hem.values,var_anoms_hem.lat.values)

                    if hemisphere=='s':
                        eofs = eofs[:,:,::-1]
                        eofs_500 = eofs_500[:,::-1]
                        eofs_va = eofs_va[:,::-1]
                        orig_var_hem_store = orig_var_hem.values[:,::-1]                                
                    else:
                        orig_var_hem_store = orig_var_hem.values                

                    eof_ds[f'{eof_var}_EOFs_{hemisphere}_{time_frame}'] = (('eof_num', 'pfull', 'lat'), eofs) 

                    eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'] = ((time_dim_name, 'eof_num'), pc1)     

                    eof_ds[f'{eof_var}_var_frac_{hemisphere}_{time_frame}'] = (('eof_num'), variance_fractions)     
                    eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'] = (('pfull', 'lat'), orig_var_hem_store) 


                    eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'] = eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'].transpose('eof_num', time_dim_name)

                    #now for 500hPa
                    eof_ds[f'{eof_var}_500_EOFs_{hemisphere}_{time_frame}'] = (('eof_num', 'lat'), eofs_500) 

                    eof_ds[f'{eof_var}_500_PCs_{hemisphere}_{time_frame}'] = ((time_dim_name, 'eof_num'), pc1_500)     

                    eof_ds[f'{eof_var}_500_var_frac_{hemisphere}_{time_frame}'] = (('eof_num'), variance_fractions_500)     

                    eof_ds[f'{eof_var}_500_PCs_{hemisphere}_{time_frame}'] = eof_ds[f'{eof_var}_500_PCs_{hemisphere}_{time_frame}'].transpose('eof_num', time_dim_name)


                    if eof_var=='ucomp':
                        ucomp_solver_dict[time_frame][hemisphere] = solver
                        ucomp_500_solver_dict[time_frame][hemisphere] = solver_500
                        ucomp_va_solver_dict[time_frame][hemisphere] = solver_va


                    eof_ds[f'{eof_var}_va_EOFs_{hemisphere}_{time_frame}'] = (('eof_num', 'lat'), eofs_va) 

                    eof_ds[f'{eof_var}_va_PCs_{hemisphere}_{time_frame}'] = ((time_dim_name, 'eof_num'), pc1_va)     

                    eof_ds[f'{eof_var}_va_var_frac_{hemisphere}_{time_frame}'] = (('eof_num'), variance_fractions_va)     
                    # eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'] = (('pfull', 'lat'), orig_var_hem_store) 

                    eof_ds[f'{eof_var}_va_PCs_{hemisphere}_{time_frame}'] = eof_ds[f'{eof_var}_va_PCs_{hemisphere}_{time_frame}'].transpose('eof_num', time_dim_name)

        for eof_var in eof_vars:
            for hemisphere in ['n', 's']:
                for time_frame in all_time_season_list:

                    var_anoms = anom_ds[f'{eof_var}_anom']

                    for use_va in [True, False, 500.]:

                        var_anoms_hem = var_anoms.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)

                        va_str = ''
                        if use_va==500.:
                            ucomp_solver = ucomp_500_solver_dict[time_frame][hemisphere]
                            var_anoms_hem = var_anoms_hem.sel(pfull=500., method='nearest')
                            va_str = '_500'                           
                        elif use_va:
                            ucomp_solver = ucomp_va_solver_dict[time_frame][hemisphere]
                            var_anoms_hem = vert_integrate(var_anoms_hem)
                            va_str = '_va'                        
                        else:
                            ucomp_solver = ucomp_solver_dict[time_frame][hemisphere]
                        
                        # var_anoms_ucomp = anom_ds[f'ucomp_anom']

                        # var_anoms_hem_ucomp = var_anoms_ucomp.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)

                        if time_frame!='all_time':
                            var_anoms_hem = var_anoms_hem.where(eof_ds['time'].dt.month.isin(season_month_dict[time_frame]), drop=True) 
                            # var_anoms_hem_ucomp = var_anoms_hem_ucomp.where(eof_ds['time'].dt.month.isin(hemisphere_month_dict[hemisphere]), drop=True) 
                            time_dim_name = f'time_{time_frame}'
                        else:
                            time_dim_name = 'time'

                        # try:

                        var_anoms_hem = var_anoms_hem - var_anoms_hem.mean('time', skipna=False)
                        # try:
                        pseudo_pcs = ucomp_solver.projectField(var_anoms_hem.values, neofs=n_eofs, eofscaling=1) #eof scaling 1 to match scaling for original eofs so that projecting u' onto u eofs produces same PC as original.
                        # except ValueError:
                            # pdb.set_trace()

                        eof_ds[f'{eof_var}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'] = ((time_dim_name, 'eof_num'), pseudo_pcs)     

                        eof_ds[f'{eof_var}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'] = eof_ds[f'{eof_var}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'].transpose('eof_num', time_dim_name)

                        logging.info(f'made it for {eof_var}{va_str} {hemisphere} {time_frame}')
                        # except:
                        #     logging.info(f'projection failed for {eof_var}{va_str} {hemisphere} {time_frame}')                        

        eof_ds.to_netcdf(output_eof_file_use)   
        eof_ds.close()
        eof_ds = xar.open_mfdataset(output_eof_file_use, decode_times=True)

    return eof_ds  

def vert_integrate(data_array):
    vert_int = data_array.integrate(coord='pfull') / data_array['pfull'].integrate(coord='pfull')    

    return vert_int

def arctan_omega(xdata_in, tau):
    return np.arctan(xdata_in*tau)

def power_spectrum_analysis(eof_ds, plot_dir, season_month_dict, use_div1_proj, scaling_density=True):

    power_spec_ds = xar.Dataset()
    va_str_dict = {True:'_va', False:''}

    season_list = [season_val for season_val in season_month_dict.keys()]

    all_time_season_list = season_list+['all_time']

    for var_to_analyse in ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']:#, 'div2_QG', 'fvbarstar']:
        for hemisphere in ['n', 's']:
            for use_va in [True, False]:
                va_str = va_str_dict[use_va]
                for time_frame in all_time_season_list:
                    plot_dir_PS = f'{plot_dir}/power_spec_plots/{hemisphere}/{time_frame}/{va_str}'

                    if not os.path.isdir(plot_dir_PS):
                        os.makedirs(plot_dir_PS)                

                    if time_frame!='all_time':
                        time_name = f'time_{time_frame}'
                    else:
                        time_name = 'time'

                    if scaling_density:
                        scaling='density'
                    else:
                        scaling='spectrum'

                    if use_div1_proj:
                        div1_name = f'{var_to_analyse}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'
                    else:
                        div1_name = f'{var_to_analyse}{va_str}_PCs_{hemisphere}_{time_frame}'

                    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'

                    for ps_var in [ucomp_name, div1_name]:
                        ps_data = eof_ds[ps_var][0,...].squeeze()

                        # logging.info(f"ps data time mean = {ps_data.mean('time').values}")

                        ps_data = ps_data - ps_data.mean(time_name).values

                        # logging.info(f"ps data time AFTER mean = {ps_data.mean('time').values}")

                        ps_data_values = ps_data.values

                        power_spec_ds = compute_segmented_power_spectrum(ps_data, scaling, power_spec_ds, dim=time_name)
                        epf.plot_power_spectrum(power_spec_ds, ps_var, va_str, plot_dir_PS, scaling, time_name)

                    cospec_data1 = eof_ds[ucomp_name][0,...].squeeze()
                    cospec_data2 = eof_ds[div1_name][0,...].squeeze()

                    power_spec_ds = compute_cospectrum(cospec_data1, cospec_data2, power_spec_ds, ucomp_name = ucomp_name, div1_name = div1_name, dim=time_name)

                    power_spec_ds = compute_phase_difference(power_spec_ds, div1_name, ucomp_name, time_name)

                    epf.plot_coherence_cospectrum_phase_diff(power_spec_ds, ucomp_name, div1_name, plot_dir_PS, va_str, scaling, time_name)   
                    plt.close('all')


    for hemisphere in ['n', 's']:
        for use_va in [True, False]:
            va_str = va_str_dict[use_va]

            plot_dir_PS = f'{plot_dir}/power_spec_plots/{hemisphere}/{time_frame}/{va_str}'

            if not os.path.isdir(plot_dir_PS):
                os.makedirs(plot_dir_PS)       

            var_list = ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']

            epf.plot_multiple_power_spectra(var_list, power_spec_ds, ps_var, va_str, plot_dir_PS, scaling, use_div1_proj, hemisphere, time_frame, time_name)
            plt.close('all')

    power_spec_ds.to_netcdf(f'{plot_dir}/power_spec.nc', auto_complex=True)


def b_fit_simpson_2013(eof_ds, plot_dir, season_month_dict, use_div1_proj):

    b_dataset = xar.Dataset()

    b_dataset.coords['lag'] = np.arange(15)
    va_str_dict = {True:'_va', False:'', 500.:'_500'}

    season_list = [season_val for season_val in season_month_dict.keys()]

    all_time_season_list = season_list+['all_time']


    for hemisphere in ['n', 's']:#, 's_DJF']:
        for use_va in [True, False, 500.]:
            va_str = va_str_dict[use_va]   
            for time_frame in all_time_season_list:
                plot_dir_b = f'{plot_dir}/b_plots/{hemisphere}_hemisphere/{time_frame}/{va_str}'

                if not os.path.isdir(plot_dir_b):
                    os.makedirs(plot_dir_b)

                plt.figure()
                for var_to_analyse in ['div1_QG', 'div1_QG_123', 'div1_QG_gt3']:

                    if use_div1_proj:
                        div1_name = f'{var_to_analyse}{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}'
                    else:
                        div1_name = f'{var_to_analyse}{va_str}_PCs_{hemisphere}_{time_frame}'    

                    ucomp_name = f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}'

                    ntime = eof_ds.coords['time'].shape[0]

                    if time_frame!='all_time':
                        where_hem = np.where(eof_ds['time'].dt.month.isin(season_month_dict[time_frame])) 

                        eddy_data = np.zeros((ntime))+np.nan
                        zonal_wind_data = np.zeros((ntime))+np.nan

                        eddy_data[where_hem[0]]       =  eof_ds[div1_name][0,:].values
                        zonal_wind_data[where_hem[0]] =  eof_ds[ucomp_name][0,:].values
                    else:
                        eddy_data = eof_ds[div1_name][0,...].squeeze().values
                        zonal_wind_data = eof_ds[ucomp_name][0,...].squeeze().values         

                    b_arr = np.zeros(15) + np.nan

                    for lag_length in range(7,15):

                        x_data = zonal_wind_data[0:-lag_length]
                        y_eddy_data = eddy_data[lag_length:]
                        y_zonal_wind_data = zonal_wind_data[lag_length:]

                        eddy_valid_mask = np.logical_and(np.isfinite(x_data), np.isfinite(y_eddy_data))
                        zonal_wind_valid_mask = np.logical_and(np.isfinite(x_data), np.isfinite(y_zonal_wind_data))                        
                        try:
                            eddy_lag_regress = scipy.stats.linregress(x_data[eddy_valid_mask], y_eddy_data[eddy_valid_mask])
                            wind_lag_regress = scipy.stats.linregress(x_data[zonal_wind_valid_mask], y_zonal_wind_data[zonal_wind_valid_mask])
                        except:
                            pdb.set_trace()

                        b_arr[lag_length] = eddy_lag_regress.slope/wind_lag_regress.slope

                    b_dataset[f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'] = (('lag'), b_arr)
                    b_av = b_dataset[f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'].mean('lag').values
                    b_dataset[f'ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}_{time_frame}'].plot(label=f'{var_to_analyse}{va_str} av = {b_av}')
                
                ax = plt.gca()
                # Move x-axis to y=0
                ax.spines['bottom'].set_position(('data', 0))

                # Hide top and right spines for a cleaner look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)            
                plt.ylim((-0.1, 0.1))
                plt.legend()
                plt.savefig(f'{plot_dir_b}/ucomp{va_str}_{var_to_analyse}{va_str}_b_{hemisphere}.pdf')

    b_dataset.to_netcdf(f'{plot_dir}/b_dataset.nc')

    b_dataset.close()

    b_dataset = xar.open_dataset(f'{plot_dir}/b_dataset.nc')

    return b_dataset

def daily_average(dataset, output_file, force_day_av_recalculate, monthly_too=False, monthly_output_file=None):
    '''This function takes a daily average of the entire dataset, including
    variables we read in at the start. This is somewhat inefficient, as we'll
    end up calculating the daily average of u, v, t etc, which we already have daily
    average data for in the cases of JRA-55 etc. However, we do this to maintain consistency between how the daily average of the eddy fluxes are calculated and how the other daily averages are calculated.'''

    if os.path.isfile(output_file) and not force_day_av_recalculate:
        logging.info('attempting to read in daily av data')
        time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
        dataset_daily = xar.open_mfdataset(output_file, decode_times=time_coder, engine='netcdf4')
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated daily av data - CALCULATING')    

        #Here we want to average over each day individually from the 6hourly snapshot data we've been given. Therefore we can use the resample function to group the data points into each day in the dataset, and then mean. We have decided not to use xcdat's temporal functions here as they require time bounds, which we are specifically not using because it's snapshot data and not daily averages etc. 

        dataset_daily = dataset.resample(time='1D').mean('time')

        if 'time_bnds' not in dataset_daily.variables.keys():

            if dataset_daily.time.encoding=={}:
                dataset_daily.time.encoding['calendar'] = dataset.time.encoding['calendar']
                dataset_daily.time.encoding['units'] = dataset.time.encoding['units']                
                dataset_daily.time.encoding['dtype'] = dataset.time.encoding['dtype']                
            dataset_daily = dataset_daily.bounds.add_missing_bounds(axes='T') #makes sure time bounds are present, as otherwise the departures method does not work. Axes='T' here refers to the time axis.     

        dataset_daily.to_netcdf(output_file)

    if monthly_too:
        if os.path.isfile(monthly_output_file) and not force_day_av_recalculate:
            logging.info('attempting to read in monthly av data')
            time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
            dataset_monthly = xar.open_mfdataset(monthly_output_file, decode_times=time_coder, engine='netcdf4')
            logging.info('SUCCESS')
        else:
            logging.info('failed to read in previously calculated monthly av data - CALCULATING')    

            year_values = dataset_daily.time.dt.year.values
            unique_year_values = np.unique(year_values)
            years_to_remove = []
            for year_val in unique_year_values:
                n_year = np.where(dataset_daily.time.dt.year == year_val)[0].shape[0]
                if n_year==1:
                    years_to_remove.append(year_val)

            if len(years_to_remove)==1:
                dataset_daily = dataset_daily.where(dataset_daily.time.dt.year!=years_to_remove[0], drop=True)
                logging.info(f'Removed single year {years_to_remove[0]} as only one day from this year - continuing')

            elif len(years_to_remove)>1:
                raise NotImplementedError('Help')
            else:
                logging.info('No single dates to remove - continuing')

            dataset_monthly = dataset_daily.resample(time='1ME').mean('time')

            if 'time_bnds' not in dataset_monthly.variables.keys():

                if dataset_monthly.time.encoding=={}:
                    dataset_monthly.time.encoding['calendar'] = dataset_daily.time.encoding['calendar']
                    dataset_monthly.time.encoding['units'] = dataset_daily.time.encoding['units']                
                    dataset_monthly.time.encoding['dtype'] = dataset_daily.time.encoding['dtype']                
                dataset_monthly = dataset_monthly.bounds.add_missing_bounds(axes='T') #makes sure time bounds are present, as otherwise the departures method does not work. Axes='T' here refers to the time axis.     

            dataset_monthly.to_netcdf(monthly_output_file)

            logging.info('Monthly output complete')    


    return dataset_daily

def read_daily_averages(yearly_data_dir, start_month, end_month, daily_monthly='daily'):

    logging.info(f'reading {daily_monthly} average files')
    daily_av_files = []

    if type(start_month)!=list:
        for year_val in range(start_month, end_month+1):
            file_name = f'{yearly_data_dir}/{year_val}_{daily_monthly}_averages.nc'
            daily_av_files = daily_av_files + [file_name]
    else:
        for year_idx, start_date_val in tqdm(enumerate(start_month)):    

            end_date_val = end_month[year_idx]
            file_name = f'{yearly_data_dir}/{start_date_val}_{end_date_val}_{daily_monthly}_averages.nc'
            daily_av_files = daily_av_files + [file_name]        

    time_coder = xar.coders.CFDatetimeCoder(use_cftime=True)
    # pdb.set_trace()
    try:
        dataset = xar.open_mfdataset(daily_av_files, decode_times=time_coder,
                                parallel=False)
    except ValueError:
        logging.warning('suspecting problem with pfull axes being different across datasets')
        dataset_list = [xar.open_mfdataset(daily_av_files[dataset_idx], decode_times=time_coder,
                                parallel=False) for dataset_idx in range(len(daily_av_files))]
        
        dataset_subset_pfull = [ds.sel(pfull=dataset_list[0].pfull.values) for ds in dataset_list]

        logging.warning('now merging two datasets')
        dataset = xar.merge(dataset_subset_pfull, compat='override')
        logging.warning('finished merging two datasets')

    logging.info(f'finished reading {daily_monthly} average files')

    dataset, duplicates_found = check_for_duplicate_times(dataset)

    if duplicates_found:
        dataset, duplicates_found_2 = check_for_duplicate_times(dataset) #run a second time to check if successful
        assert(not duplicates_found_2) #make sure that duplicates were not found a second time, if not throw error

    return dataset

def check_for_duplicate_times(dataset):

    # time_diff = dataset.time.diff(dim='time')
    # where_duplicates = np.where(time_diff!=time_diff[0])
    n_unique_times = np.unique(dataset.time).shape[0]
    n_times = dataset.time.shape[0]

    if n_times>n_unique_times:
        is_duplicates = True
    else:
        is_duplicates = False

    n_time_orig = dataset.time.shape[0]

    if is_duplicates:
        logging.info(f'duplicate times are present!!!')
        _, unique_time_indexes = np.unique(dataset.time, return_index=True)
        dataset = dataset.isel(time=unique_time_indexes)
        n_time_new = dataset.time.shape[0]        
        logging.info(f'duplicate times removed. Was {n_time_orig} now {n_time_new}')
        duplicates_found = True
    else:
        logging.info(f'duplicate times are NOT present - continuing')
        duplicates_found = False
    
    return dataset, duplicates_found