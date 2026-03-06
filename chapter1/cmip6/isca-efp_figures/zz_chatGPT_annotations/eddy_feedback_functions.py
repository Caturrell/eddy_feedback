import numpy as np
from eofs.standard import Eof
from tqdm import tqdm
from scipy import optimize as scipyo
import scipy
import statsmodels.tsa.stattools as sm 
import scipy.signal as signal 
import xarray as xar
import os
import aostools.climate as aoscli
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import xcdat

import pdb  # Python debugger; used for debugging purposes

def compute_segmented_power_spectrum(data, scaling, power_spec_ds, dim='time', segment_length=256, overlap=128):
    """
    Compute the power spectrum using overlapping segments and a Hanning window.

    Parameters:
        data (xarray.DataArray): Input time series.
        scaling (str): Scaling parameter for the power spectrum ('density' or 'spectrum').
        power_spec_ds (xarray.Dataset): Dataset to which computed spectra will be added.
        dim (str): The time dimension (default: 'time').
        segment_length (int): Length of each segment in time units (e.g., days).
        overlap (int): Overlapping length between segments (e.g., days).

    Returns:
        xarray.Dataset: Updated dataset including frequency coordinate and computed spectra.
    """
    # Get time values and compute time step (in seconds)
    time_values = data[dim].values
    dt = np.median(np.diff(time_values)) / np.timedelta64(1, 's')  # Compute median time difference in seconds
    
    # Check if the time step corresponds to daily data (86400 seconds)
    if np.isclose(dt, 86400.):
        dt = 1.  # Set dt to 1 day for simplicity
    else:
        raise NotImplemented('needs to be daily data')    

    # Calculate sampling frequency in cycles per day
    fs = 1 / dt

    # Convert segment length and overlap from days to number of data points
    segment_size = int(segment_length / dt)
    overlap_size = int(overlap / dt)

    # Compute power spectrum using Welch’s method
    freqs, power_spectrum = signal.welch(
        data.values, fs=fs, 
        window='hann',  # Use Hanning window to reduce spectral leakage
        nperseg=segment_size, 
        noverlap=overlap_size,
        scaling=scaling
    )

    # Compute Short-Time Fourier Transform (STFT) for further spectral analysis
    freqs2, t, complex_spectrum = signal.stft(data.values, fs=fs, window='hann', nperseg=segment_size, noverlap=overlap_size, scaling='psd') 

    # Compute amplitude spectrum (power) from the complex STFT coefficients
    complex_spectrum_amp = np.abs(complex_spectrum) ** 2.

    # Extract dimension names from the input dataset to check for existing coordinates
    dim_names = [key for key in power_spec_ds.coords.keys()]

    # Add 'frequency' coordinate if it does not exist in power_spec_ds
    if 'frequency' not in dim_names:
        power_spec_ds.coords['frequency'] = (('frequency'), freqs)

    # Add 't' coordinate for STFT time points if it does not exist
    if 't' not in dim_names:
        power_spec_ds.coords['t'] = (('t'), t)

    # Save Welch power spectrum into the dataset with a name based on data.name
    power_spec_ds[f'{data.name}_power_spec_welch'] = (('frequency'), power_spectrum)

    # Save the mean STFT power spectrum into the dataset
    power_spec_ds[f'{data.name}_power_spec_stft'] = (('frequency'), np.mean(complex_spectrum_amp, axis=1))

    # Save the full complex STFT coefficients into the dataset
    power_spec_ds[f'{data.name}_fourier_coeffs_stft'] = (('frequency', 't'), complex_spectrum)

    return power_spec_ds

def compute_cospectrum(data1, data2, power_spec_ds, dim='time', segment_length=256, overlap=128, ucomp_name='ucomp_s_all_time', div1_name=''):
    """
    Compute the cospectrum between two time series using Welch's method.

    Parameters:
        data1 (xarray.DataArray): First input time series.
        data2 (xarray.DataArray): Second input time series.
        power_spec_ds (xarray.Dataset): Dataset that already contains power spectrum information.
        dim (str): The time dimension (default: 'time').
        segment_length (int): Length of each segment in time units.
        overlap (int): Overlapping length between segments.
        ucomp_name (str): Variable name for the first series (used for normalization).
        div1_name (str): Variable name for the second series (used in naming outputs).

    Returns:
        xarray.Dataset: Updated dataset including cospectrum and coherence metrics.
    """
    # Ensure that both time series share the same time coordinate
    if not np.array_equal(data1[dim], data2[dim]):
        raise ValueError("The two time series must have the same time coordinate.")

    # Calculate time step from the time coordinate
    time_values = data1[dim].values
    dt = np.median(np.diff(time_values)) / np.timedelta64(1, 's')  # in seconds

    if np.isclose(dt, 86400.):
        dt = 1.
    else:
        raise NotImplemented('needs to be daily data')    

    # Sampling frequency in cycles per day
    fs = 1 / dt

    # Convert segment parameters from days to number of points
    segment_size = int(segment_length / dt)
    overlap_size = int(overlap / dt)

    # Compute cross-power spectral density (CSD) using Welch's method
    freqs, csd = signal.csd(
        data1.values, data2.values, fs=fs, 
        window='hann',
        nperseg=segment_size, 
        noverlap=overlap_size,
        scaling='density'
    )

    # Retrieve the power spectrum of the u-component from the dataset
    power_spectrum_ucomp_da = power_spec_ds[f'{ucomp_name}_power_spec_welch'].values

    # Normalize the cross-spectral density by the u-component power spectrum
    csd = csd / power_spectrum_ucomp_da

    # Extract the cospectrum (real part) and quadrature spectrum (imaginary part)
    cospectrum = np.real(csd)
    cospectrum_im = np.imag(csd)

    # Compute STFT-based cross-spectral density (unnormalized)
    csd_stft_unnorm = np.mean(np.conjugate(power_spec_ds[f'{ucomp_name}_fourier_coeffs_stft']) * power_spec_ds[f'{div1_name}_fourier_coeffs_stft'], axis=1)

    # Normalize the STFT-based CSD using the STFT power spectrum of the u-component
    csd_stft = csd_stft_unnorm / power_spec_ds[f'{ucomp_name}_power_spec_stft']

    # Compute coherence using Welch's method
    freqs_co, coher = signal.coherence(data1.values, data2.values, fs=fs, window='hann', nperseg=segment_size, noverlap=overlap_size)

    # Compute STFT-based coherence
    coher_stft = np.abs(csd_stft_unnorm)**2. / (power_spec_ds[f'{ucomp_name}_power_spec_stft'] * power_spec_ds[f'{div1_name}_power_spec_stft'])

    # Save the computed Welch-based cospectrum and coherence into the dataset
    power_spec_ds[f'{data1.name}_{data2.name}_cospec_welch'] = (('frequency'), csd)
    power_spec_ds[f'{data1.name}_{data2.name}_coher_welch'] = (('frequency'), coher)

    # Save the computed STFT-based cospectrum and coherence into the dataset
    power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft'] = (('frequency'), csd_stft.values)
    power_spec_ds[f'{data1.name}_{data2.name}_coher_stft'] = (('frequency'), coher_stft.values)

    return power_spec_ds

def compute_phase_difference(power_spec_ds, div1_name, ucomp_name):
    """
    Compute the phase difference between two series using STFT coefficients and perform curve fitting to estimate timescales.

    Parameters:
        power_spec_ds (xarray.Dataset): Dataset that contains Fourier coefficients and other spectral data.
        div1_name (str): Name of the div1 variable.
        ucomp_name (str): Name of the ucomp variable.

    Returns:
        xarray.Dataset: Updated dataset including phase differences and fitted timescales.
    """
    # Calculate the mean phase difference (in degrees) from the STFT Fourier coefficients
    div1_ucomp_phase_diff = np.mean(
        np.angle(power_spec_ds[f'{div1_name}_fourier_coeffs_stft'] / power_spec_ds[f'{ucomp_name}_fourier_coeffs_stft'], deg=True),
        axis=1
    )
    # Convert frequency to angular frequency (omega = 2*pi*f)
    omega_freqs = 2. * np.pi * power_spec_ds['frequency']

    # Fit the arctan function to the phase difference data (in radians)
    timescale_from_phase_diff, _ = scipyo.curve_fit(arctan_omega, omega_freqs.values, np.deg2rad(div1_ucomp_phase_diff))
    timescale_from_phase_diff_days = timescale_from_phase_diff[0]  # Timescale (in days)

    # Limit the fitting to low frequency points (<=0.025 cycles per day)
    where_low_freq = np.where(power_spec_ds['frequency'].values <= 0.025)[0]
    logging.info(f'Using only {np.shape(where_low_freq)[0]} points for fit of tau')

    omega_freqs_lim = omega_freqs[where_low_freq]
    div1_ucomp_phase_diff_lim = div1_ucomp_phase_diff[where_low_freq]   

    # Perform a second curve fitting using the limited frequency range
    timescale_from_phase_diff2, _ = scipyo.curve_fit(arctan_omega, omega_freqs_lim.values, np.deg2rad(div1_ucomp_phase_diff_lim))
    timescale_from_phase_diff_days2 = timescale_from_phase_diff2[0]

    # Save the phase difference and fitted timescales into the dataset
    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff'] = (('frequency'), div1_ucomp_phase_diff)
    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_1'] = timescale_from_phase_diff_days
    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_2'] = timescale_from_phase_diff_days2

    # Perform a linear regression on the real and imaginary parts of the STFT cospectrum at low frequencies
    real_cospec_stft = np.real(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft'])
    imag_cospec_stft = np.imag(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stht'])
    lin_reg_real = scipy.stats.linregress(omega_freqs[where_low_freq], real_cospec_stft[where_low_freq])
    lin_reg_imag = scipy.stats.linregress(omega_freqs[where_low_freq], imag_cospec_stft[where_low_freq])    

    # Estimate tau from the ratio of the imaginary slope to the real intercept
    tau_estimate_3 = lin_reg_imag.slope / lin_reg_real.intercept
    power_spec_ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'] = tau_estimate_3

    return power_spec_ds

def eof_calc_alt(data, lats):
    """
    Calculate EOFs (Empirical Orthogonal Functions) on the input data with latitude weighting.

    Parameters:
        data (numpy.ndarray): Input data array (e.g., anomalies).
        lats (numpy.ndarray): Array of latitudes corresponding to the data.

    Returns:
        tuple: A tuple containing the EOFs (as covariance), principal components (PC1), variance fractions, and the solver object.
    """
    # Compute cosine of latitudes for weighting and ensure no negative weights
    coslat = np.cos(np.deg2rad(lats)).clip(0., 1.)
    wgts = np.sqrt(coslat)[np.newaxis, np.newaxis, :]

    # Create an EOF solver with the given weights and center the data
    solver = Eof(data, weights=wgts, center=True)

    # Compute the first three EOFs as covariance patterns
    eofs = solver.eofsAsCovariance(neofs=3)
    # Compute the principal components (time series) associated with the EOFs
    pc1 = solver.pcs(npcs=3, pcscaling=1)
    # Compute the fraction of variance explained by each EOF
    variance_fractions = solver.varianceFraction(neigs=3)

    return eofs, pc1, variance_fractions, solver

def cross_correlation(series1, series2, max_lag):
    """
    Compute the cross-correlation between two time series for lags ranging from -max_lag to +max_lag.

    Parameters:
        series1 (numpy.ndarray): First time series.
        series2 (numpy.ndarray): Second time series.
        max_lag (int): Maximum lag (number of time steps) to consider.

    Returns:
        tuple: A tuple containing the correlation coefficients and corresponding lags.
    """
    # Create an array of lag values
    lags = np.arange(-max_lag, max_lag + 1)
    len_series1 = series1.shape[0]
    len_series2 = series2.shape[0]
    assert(len_series1 == len_series2)

    # Define index range that avoids boundary effects when shifting the series
    series1_start_idx = max_lag
    series1_end_idx   = len_series1 - max_lag

    corr = []
    # Loop over each lag to compute the correlation coefficient
    for lag in lags:
        # Slice the first series using the defined indices
        series_1_short = series1[series1_start_idx:series1_end_idx]
        # Slice the second series with a shift corresponding to the current lag
        series_2_short = series2[max_lag + lag:len_series2 - max_lag + lag]

        # Identify indices where both series have finite values (exclude NaNs)
        valid_idx = np.isfinite(series_1_short) & np.isfinite(series_2_short)

        # Compute the correlation coefficient for valid data points
        corr.append(np.corrcoef(series_1_short[valid_idx], series_2_short[valid_idx])[0, 1])

    return corr, lags

def sm_cross_correlation(series1, series2, max_lag):
    """
    Compute the cross-correlation using the statsmodels implementation.

    Parameters:
        series1 (numpy.ndarray): First time series.
        series2 (numpy.ndarray): Second time series.
        max_lag (int): Maximum lag to compute.

    Returns:
        numpy.ndarray: Cross-correlation function values.
    """
    # Use statsmodels' ccf function, which computes cross-correlation for nlags=max_lag
    ccf_values = sm.ccf(series1, series2, nlags=max_lag)
    return ccf_values

def ep_flux_calc(dataset, output_file, force_ep_flux_recalculate, include_udt_rdamp, omega_rot_rate, a0):
    """
    Calculate EP flux components and related variables from the dataset.

    Parameters:
        dataset (xarray.Dataset): Input dataset with necessary fields.
        output_file (str): File path to save the computed EP flux data.
        force_ep_flux_recalculate (bool): Flag to force recalculation even if output_file exists.
        include_udt_rdamp (bool): Flag indicating whether to include additional damping terms.
        omega_rot_rate (float): Rotational rate (e.g., Earth's rotation rate).
        a0 (float): Reference radius (e.g., Earth's radius).

    Returns:
        xarray.Dataset: Dataset containing the EP flux and associated fields.
    """
    # Check if the output file already exists and if recalculation is not forced
    if os.path.isfile(output_file) and not force_ep_flux_recalculate:
        logging.info('attempting to read in data')
        epflux_ds = xar.open_mfdataset(output_file, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated data - CALCULATING')

        # Get list of variable names in the dataset
        dataset_vars = [key for key in dataset.variables.keys()]
        omega_present = 'omega' in dataset_vars

        # Create a new xarray Dataset with the same coordinates as the input dataset
        epflux_ds = xar.Dataset(coords=dataset.coords)

        logging.info('calculating ep flux')
        if omega_present:
            # Calculate EP flux components using the provided function from aostools
            epflux_ds['ep1'], epflux_ds['ep2'], epflux_ds['div1'], epflux_ds['div2'], epflux_ds['dthdp_bar'] = aoscli.ComputeEPfluxDivXr(
                dataset['ucomp'], dataset['vcomp'], dataset['temp'],
                do_ubar=True, w=dataset['omega'] / 100., ref='instant'
            )  # Note: omega is divided by 100 to convert from Pa/s to hPa/s.
        
        logging.info('calculating QG ep flux')
        # Calculate quasi-geostrophic (QG) EP flux components
        epflux_ds['ep1_QG'], epflux_ds['ep2_QG'], epflux_ds['div1_QG'], epflux_ds['div2_QG'], epflux_ds['dthdp_bar_QG'] = aoscli.ComputeEPfluxDivXr(
            dataset['ucomp'], dataset['vcomp'], dataset['temp'],
            do_ubar=False, ref='instant'
        )

        # If omega is not present, use QG-derived dthdp_bar
        if not omega_present:
            epflux_ds['dthdp_bar'] = epflux_ds['dthdp_bar_QG']

        # Compute the inverse of dthdp_bar (used in further calculations)
        epflux_ds['inv_dthdp_bar'] = 1. / epflux_ds['dthdp_bar']

        logging.info('preparing data for vbarstar calc')
        logging.info('calculating vbarstar')
        # Calculate vbarstar using a function from aostools; this uses the entire dataset
        epflux_ds['vbarstar'] = aoscli.ComputeVstar(dataset, ref='instant')

        # Compute fvbarstar as the product of vbarstar, 2*omega, and sine of latitude
        epflux_ds['fvbarstar'] = epflux_ds['vbarstar'] * 2. * omega_rot_rate * np.sin(np.deg2rad(dataset['lat']))

        if omega_present:
            logging.info('calculating omegabarstar')
            # Calculate omegabarstar using aostools; again, convert omega units as needed
            epflux_ds['omegabarstar'] = aoscli.ComputeWstarXr(
                dataset['omega'] / 100., dataset['temp'], dataset['vcomp'], pres='pfull', ref='instant'
            )

        logging.info('calculating other terms')
        # Compute the gradient in the meridional (latitude) direction (in radians)
        dphi = np.gradient(np.deg2rad(dataset['lat']).values)[np.newaxis, np.newaxis, :]

        # Compute the meridional gradient of u (weighted by cosine of latitude)
        dudphi = aoscli.FlexiGradPhi(dataset['ucomp'].mean('lon') * np.cos(np.deg2rad(dataset['lat'])).values, dphi)
        epflux_ds['dudphi'] = (('time', 'pfull', 'lat'), dudphi)

        # Scale the gradient by 1/(a0*cos(latitude))
        epflux_ds['1oacosphi_dudphi'] = epflux_ds['dudphi'] * (1. / (a0 * np.cos(np.deg2rad(dataset['lat']))))

        # Compute the pressure gradient; note that pressure is converted to Pa
        dp = np.gradient(dataset['pfull'].values * 100.)[np.newaxis, :, np.newaxis]

        # Compute the vertical gradient of u
        dudp = aoscli.FlexiGradP(dataset['ucomp'].mean('lon').values, dp)
        epflux_ds['dudp'] = (('time', 'pfull', 'lat'), dudp)

        # Compute a term combining vbarstar and the scaled meridional gradient of u
        epflux_ds['vbarstar_1oacosphi_dudphi'] = (epflux_ds['vbarstar'] * epflux_ds['1oacosphi_dudphi'])

        if omega_present:
            # Compute the product of omegabarstar and dudp
            epflux_ds['omegabarstar_dudp'] = epflux_ds['omegabarstar'] * epflux_ds['dudp']
            # Combine terms to compute the total tendency (with unit conversion from seconds to days)
            epflux_ds['total_tend'] = (
                epflux_ds['fvbarstar'] - epflux_ds['vbarstar_1oacosphi_dudphi'] - epflux_ds['omegabarstar_dudp']
                + epflux_ds['div1'] / 86400. + epflux_ds['div2'] / 86400.
            )

        # Compute the QG version of the total tendency
        epflux_ds['total_tend_QG'] = epflux_ds['fvbarstar'] + epflux_ds['div1_QG'] / 86400. + epflux_ds['div2_QG'] / 86400.

        # Optionally include an additional damping term if specified
        if include_udt_rdamp:
            epflux_ds['total_tend'] = epflux_ds['total_tend'] + dataset['udt_rdamp'].mean('lon')
            epflux_ds['total_tend_QG'] = epflux_ds['total_tend_QG'] + dataset['udt_rdamp'].mean('lon')

        # Compute the time derivative of the zonal mean u (difference between first and last time point)
        epflux_ds['delta_ubar_dt'] = (
            (dataset['ucomp'][-1, ...].mean('lon').squeeze() - dataset['ucomp'][0, ...].mean('lon').squeeze())
            / (86400. * (dataset['time'][-1].values - dataset['time'][0].values).days)
        )

        logging.info('writing EP flux data etc to file')
        epflux_ds.to_netcdf(output_file)
        logging.info('FINISHED writing EP flux data etc to file')

        epflux_ds.close()
        epflux_ds = xar.open_mfdataset(output_file, decode_times=True)

    return epflux_ds

def efp_calc(output_efp_file, force_efp_recalculate, dataset, vars_to_correlate, exp_type):
    """
    Calculate EFP (eddy-feedback parameter) values based on correlations between variables.

    Parameters:
        output_efp_file (str): File path to save/read the EFP data.
        force_efp_recalculate (bool): Flag to force recalculation if True.
        dataset (xarray.Dataset): Input dataset with necessary fields.
        vars_to_correlate (list): List of variable names to be correlated.
        exp_type (str): Experiment type; used to determine data slicing.

    Returns:
        xarray.Dataset: Dataset containing computed EFP values.
    """
    if os.path.isfile(output_efp_file) and not force_efp_recalculate:
        logging.info('attempting to read in EFP data')
        efp_output_ds = xar.open_mfdataset(output_efp_file, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated EFP data - CALCULATING')
        nh_winter_season = 'djf'

        # Determine start and end years from the time coordinate
        start_year = dataset.time.dt.year[0].values
        end_year = dataset.time.dt.year[-1].values
        if exp_type != 'isca':
            dataset_cut_ends = dataset.sel(time=slice(f'{start_year}-03', f'{end_year}-11'))
        else:
            dataset_cut_ends = dataset

        # Resample data seasonally (quarterly starting in December)
        seasonal = dataset_cut_ends.resample(time='QS-DEC').mean('time')

        # Define representative months for each season
        season_months = {
            'djf': 12,
            'mam': 3,
            'jja': 6,
            'son': 9
        }

        # Select DJF season for northern hemisphere winter
        seasonal_djf = seasonal.sel(time=seasonal.time.dt.month == season_months[nh_winter_season])

        # For southern hemisphere, select JAS (July, August, September)
        seasonal_jas = dataset.sel(time=dataset.time.dt.month.isin([7, 8, 9]))
        seasonal_jas = seasonal_jas.groupby('time.year').mean('time').rename({'year': 'time'})

        # Initialize a new dataset to store EFP results with common coordinates
        efp_output_ds = xar.Dataset()
        efp_output_ds.coords['lat'] = (('lat'), dataset['lat'].values)
        efp_output_ds.coords['pfull'] = (('pfull'), dataset['pfull'].values)

        # Loop over hemispheres to compute correlations
        for hemisphere in ['n', 's']:
            # Determine latitude slice based on the order of latitude values
            if np.all(dataset.lat.diff('lat').values > 0.):
                if hemisphere == 'n':                
                    efp_lat_slice = slice(25., 75.)
                else:
                    efp_lat_slice = slice(-75., -25.)
            else:
                if hemisphere == 'n':
                    efp_lat_slice = slice(75., 25.)
                else:
                    efp_lat_slice = slice(-25., -75.)
   
            # Determine pressure slice based on ordering of pressure levels
            if np.all(dataset.pfull.diff('pfull').values > 0.):
                pfull_slice = slice(200., 600.)
            else:
                pfull_slice = slice(600., 200.)

            # Select seasonal data based on hemisphere
            if hemisphere == 'n':
                efp_ds = seasonal_djf
            else:
                efp_ds = seasonal_jas

            # Loop over pairs of variables to correlate (including 'ucomp')
            for var2_to_correlate in vars_to_correlate + ['ucomp']:
                for var_to_correlate in vars_to_correlate + ['ucomp']:
                    corr_var_name = f'{var_to_correlate}_{var2_to_correlate}_{hemisphere}_corr'
                    opposite_corr_var_name = f'{var2_to_correlate}_{var_to_correlate}_{hemisphere}_corr'

                    eof_output_ds_vars = [key for key in efp_output_ds.variables.keys()]

                    # If the opposite correlation variable exists, use it to avoid duplicate computation
                    if opposite_corr_var_name in eof_output_ds_vars:
                        logging.info(f'skipping efp for {corr_var_name} as already present by a different name')
                        efp_output_ds[corr_var_name] = efp_output_ds[opposite_corr_var_name]
                        efp_output_ds[f'efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}'] = efp_output_ds[f'efp_{var2_to_correlate}_{var_to_correlate}_{hemisphere}']                     
                    else:
                        logging.info(f'calculating efp for {corr_var_name}')
                        # Get the data for the first variable and average over longitude if present
                        data_to_correlate = efp_ds[var_to_correlate]
                        if 'lon' in data_to_correlate.dims:
                            data_to_correlate = data_to_correlate.mean('lon')
                        # Get the data for the second variable and average over longitude if present
                        data2_to_correlate = efp_ds[var2_to_correlate]
                        if 'lon' in data2_to_correlate.dims:
                            data2_to_correlate = data2_to_correlate.mean('lon')
                        # Compute squared correlation coefficient over time
                        efp_output_ds[corr_var_name] = xar.corr(data_to_correlate, data2_to_correlate, dim='time') ** 2
                        # Select a slice of the correlation data in latitude and pressure
                        corr_slice = efp_output_ds[corr_var_name].sel(lat=efp_lat_slice, pfull=pfull_slice)
                        take_level_mean = True
                        if take_level_mean:
                            corr_av = corr_slice.mean('pfull')
                        else:
                            corr_av = corr_slice
                        # Weight the correlation by the cosine of latitude and average over latitude
                        weights = np.cos(np.deg2rad(corr_av.lat))
                        efp = corr_av.weighted(weights).mean('lat')
                        # Save the computed EFP value into the dataset
                        efp_output_ds[f'efp_{var_to_correlate}_{var2_to_correlate}_{hemisphere}'] = efp.values
                        logging.info(f'efp {var_to_correlate} {var2_to_correlate} = {efp.values}')

        # Write the EFP dataset to a NetCDF file and then reopen it
        efp_output_ds.to_netcdf(output_efp_file)    
        efp_output_ds.close()
        efp_output_ds = xar.open_mfdataset(output_efp_file, decode_times=True)

    return efp_output_ds

def calculate_anomalies(dataset, var_list, subtract_annual_cycle, output_anom_file, force_anom_recalculate):
    """
    Calculate anomalies for a list of variables, optionally subtracting the annual cycle.

    Parameters:
        dataset (xarray.Dataset): Input dataset containing variables.
        var_list (list): List of variable names to compute anomalies for.
        subtract_annual_cycle (bool): Flag to subtract the annual cycle.
        output_anom_file (str): File path to save the anomaly dataset.
        force_anom_recalculate (bool): Flag to force recalculation.

    Returns:
        xarray.Dataset: Dataset containing anomaly and original fields for each variable.
    """
    if os.path.isfile(output_anom_file) and not force_anom_recalculate:
        logging.info('attempting to read in anom data')
        anom_ds = xar.open_mfdataset(output_anom_file, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated anom data - CALCULATING')

        # Calculate anomalies for the first variable in the list
        var_anoms1, orig_var1 = calculate_anom_one_var(dataset, subtract_annual_cycle, var_list[0])
        anom_ds = xar.Dataset(coords=var_anoms1.coords)
        anom_ds[f'{var_list[0]}_anom'] = var_anoms1
        anom_ds[f'{var_list[0]}_orig'] = orig_var1        
    
        # Loop through the rest of the variables and compute anomalies
        for eof_var in var_list[1:]:
            var_anoms, orig_var = calculate_anom_one_var(dataset, subtract_annual_cycle, eof_var)
            anom_ds[f'{eof_var}_anom'] = var_anoms
            anom_ds[f'{eof_var}_orig'] = orig_var        
    
        # Save the anomalies dataset to file and reopen it
        anom_ds.to_netcdf(output_anom_file)   
        anom_ds.close()
        anom_ds = xar.open_mfdataset(output_anom_file, decode_times=True)

    return anom_ds

def calculate_anom_one_var(dataset, subtract_annual_cycle, data_var):
    """
    Calculate anomalies for a single variable from the dataset.

    Parameters:
        dataset (xarray.Dataset): Input dataset.
        subtract_annual_cycle (bool): Flag to subtract the annual cycle.
        data_var (str): The variable name for which to compute anomalies.

    Returns:
        tuple: A tuple of (anomaly, original variable) data arrays.
    """
    if subtract_annual_cycle:
        logging.info(f'subtracting annual cycle from {data_var}')
        var_anoms = dataset.temporal.departures(data_var=data_var, freq='day', weighted=True)[data_var].load()
        orig_var = dataset[data_var]
        # Average over longitude if present
        if 'lon' in var_anoms.dims:
            var_anoms = var_anoms.mean('lon')
            orig_var = orig_var.mean('lon')
    else:
        if 'lon' in dataset[data_var].dims:
            var_zm = dataset[data_var].mean('lon')
        else:
            var_zm = dataset[data_var]
        # Compute the time mean and subtract it
        var_zm_time = var_zm.mean('time')
        var_anoms = var_zm - var_zm_time
        orig_var = var_zm    

    return var_anoms, orig_var

def propagate_missing_data_to_all_vars(anom_ds):
    """
    Propagate missing data (NaNs) across all anomaly variables so that every field has NaNs in the same locations.

    This is necessary for the EOF projection method to work properly, ensuring consistency
    in missing data across different variables.

    Parameters:
        anom_ds (xarray.Dataset): Dataset containing anomaly fields.

    Returns:
        xarray.Dataset: Updated dataset with consistent missing data across variables.
    """
    # Get list of coordinate names and determine anomaly variable names (those containing '_anom')
    anom_coord_list = [key for key in anom_ds.coords.keys()]
    anom_var_list = [key for key in anom_ds.variables.keys() if key not in anom_coord_list and '_anom' in key]

    n_nans_list = []
    # Loop over each anomaly variable and propagate NaNs from all other anomaly variables
    for anom_var in anom_var_list:
        for anom_var_nan_var in anom_var_list:
            anom_ds[anom_var] = anom_ds[anom_var].where(np.isfinite(anom_ds[anom_var_nan_var]))
        n_nans_list.append(np.where(np.isnan(anom_ds[anom_var]))[0].shape[0])

    # Check that all anomaly variables now have the same number of NaNs
    all_anom_vars_same_number_nan = np.all(np.asarray(n_nans_list) == n_nans_list[0])
    if all_anom_vars_same_number_nan:
        logging.info(f'successfully propagated missing data from all vars to all other vars to enable eof projection, so each field now has {n_nans_list[0]} nans')
    else:
        raise ValueError('anom vars contain differing number of nans, which means the projection will not work')

    # Print a check for each pair of anomaly variables to ensure missing data locations are identical
    for anom_var in anom_var_list:
        for anom_var_nan_var in anom_var_list:
            all_same = np.all(np.isnan(anom_ds[anom_var].values) == np.isnan(anom_ds[anom_var_nan_var].values))
            print(f'{all_same} for {anom_var} and {anom_var_nan_var}')

    return anom_ds

def eof_calc(exp_type, output_eof_file, force_eof_recalculate, dataset, pfull_slice, subtract_annual_cycle, eof_vars, n_eofs, hemisphere_month_dict, anom_ds, propagate_all_nans):
    """
    Compute Empirical Orthogonal Functions (EOFs) for specified variables and save the results.

    Parameters:
        exp_type (str): Experiment type (e.g., 'held_suarez' or 'isca').
        output_eof_file (str): File path to save the EOF dataset.
        force_eof_recalculate (bool): Flag to force recalculation.
        dataset (xarray.Dataset): Input dataset with original fields.
        pfull_slice (slice): Slice object to select a range of pressure levels.
        subtract_annual_cycle (bool): Whether to subtract the annual cycle from the data.
        eof_vars (list): List of variable names for which EOFs are computed.
        n_eofs (int): Number of EOFs to compute.
        hemisphere_month_dict (dict): Dictionary mapping hemispheres to relevant months.
        anom_ds (xarray.Dataset): Dataset containing anomalies.
        propagate_all_nans (bool): Flag to propagate missing data among all anomaly variables.

    Returns:
        xarray.Dataset: Dataset containing computed EOFs, PCs, variance fractions, and time-mean fields.
    """
    # Determine hemisphere slices based on latitude order
    if np.all(dataset.lat.diff('lat') < 0.):
        hemisphere_slice_dict = {'n': slice(90., 0.), 's': slice(0., -90.)}
    else:
        hemisphere_slice_dict = {'n': slice(0., 90.), 's': slice(-90., 0.)}        

    # Modify output file name if propagation of NaNs is required
    if propagate_all_nans:
        output_eof_file_use = output_eof_file.split('.nc')[0] + '_prop_nans.nc'
    else:
        output_eof_file_use = output_eof_file

    # Check if the EOF file already exists and if recalculation is not forced
    if os.path.isfile(output_eof_file_use) and not force_eof_recalculate:
        logging.info('attempting to read in EOF data')
        eof_ds = xar.open_mfdataset(output_eof_file_use, decode_times=True)
        logging.info('SUCCESS')
    else:
        logging.info('failed to read in previously calculated EOF data - CALCULATING')
        logging.info('calculating anomalies')
        if exp_type == 'held_suarez':
            u_anoms = dataset['ucomp'].mean('lon') - dataset['ucomp'].mean(('time', 'lon'))
        else:
            u_anoms_time = dataset.temporal.departures(data_var='ucomp', freq='day', weighted=False)['ucomp'].mean('lon')

        # Select southern and northern hemisphere anomalies based on the defined slices and month criteria
        u_anoms_sh = u_anoms_time.sel(lat=hemisphere_slice_dict['s']).sel(pfull=pfull_slice).where(dataset['time'].dt.month.isin(hemisphere_month_dict['s']), drop=True)
        u_anoms_nh = u_anoms_time.sel(lat=hemisphere_slice_dict['n']).sel(pfull=pfull_slice).where(dataset['time'].dt.month.isin(hemisphere_month_dict['n']), drop=True)

        u_anoms_sh = u_anoms_sh.rename({'time': 'time_s'})

        # Create a new dataset for EOF results and set coordinates
        eof_ds = xar.Dataset(coords=u_anoms_sh.coords)
        eof_ds.coords['time_n'] = ('time_n', u_anoms_nh.time.values)
        eof_ds.coords['eof_num'] = ('eof_num', np.arange(n_eofs))

        # Initialize a dictionary to store EOF solvers for the ucomp variable
        ucomp_solver_dict = {'season': {'n': {}, 's': {}}, 'all_time': {'n': {}, 's': {}}}

        # Optionally propagate missing data to all anomaly variables
        if propagate_all_nans:
            anom_ds = propagate_missing_data_to_all_vars(anom_ds)

        # Loop over each variable for which to compute EOFs
        for eof_var in tqdm(eof_vars):
            logging.info('loading anomalies from anom_ds')
            var_anoms = anom_ds[f'{eof_var}_anom']
            orig_var = anom_ds[f'{eof_var}_orig']

            if 'time' not in eof_ds.coords.keys():
                eof_ds.coords['time'] = ('time', var_anoms.time.values)

            # Loop over hemispheres and time frames (seasonal and all time)
            for hemisphere in ['n', 's']:
                for time_frame in ['season', 'all_time']:
                    var_anoms_hem = var_anoms.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)
                    
                    # wrangle ANOMALIES
                    if time_frame == 'season':
                        var_anoms_hem = var_anoms_hem.where(eof_ds['time'].dt.month.isin(hemisphere_month_dict[hemisphere]), drop=True) 
                        time_dim_name = f'time_{hemisphere}'
                    else:
                        time_dim_name = 'time'

                    # wrangle ORIGINAL VALUES
                    orig_var_hem = orig_var.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)
                    if time_frame == 'season':
                        orig_var_hem = orig_var_hem.where(eof_ds['time'].dt.month.isin(hemisphere_month_dict[hemisphere]), drop=True)
                    orig_var_hem = orig_var_hem.mean('time')

                    logging.info(f'calculating EOFs for {eof_var}')
                    # Compute EOFs, PCs, and variance fractions using the alternative EOF function
                    eofs, pc1, variance_fractions, solver = eof_calc_alt(var_anoms_hem.values, var_anoms_hem.lat.values)

                    # For the southern hemisphere, reverse the order of latitudes in EOFs and original field
                    if hemisphere == 's':
                        eofs = eofs[:, :, ::-1]
                        orig_var_hem_store = orig_var_hem.values[:, ::-1]                                
                    else:
                        orig_var_hem_store = orig_var_hem.values                

                    # Store computed EOFs, PCs, variance fractions, and time-mean fields in the dataset
                    eof_ds[f'{eof_var}_EOFs_{hemisphere}_{time_frame}'] = (('eof_num', 'pfull', 'lat'), eofs) 
                    eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'] = ((time_dim_name, 'eof_num'), pc1)     
                    eof_ds[f'{eof_var}_var_frac_{hemisphere}_{time_frame}'] = (('eof_num'), variance_fractions)     
                    eof_ds[f'{eof_var}_mean_{hemisphere}_{time_frame}'] = (('pfull', 'lat'), orig_var_hem_store) 

                    # Transpose the PCs to have dimensions (eof_num, time)
                    eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'] = eof_ds[f'{eof_var}_PCs_{hemisphere}_{time_frame}'].transpose('eof_num', time_dim_name)

                    if eof_var == 'ucomp':
                        ucomp_solver_dict[time_frame][hemisphere] = solver

        # Loop to compute projected PCs for each variable using the ucomp EOF solver
        for eof_var in eof_vars:
            for hemisphere in ['n', 's']:
                for time_frame in ['season', 'all_time']:
                    ucomp_solver = ucomp_solver_dict[time_frame][hemisphere]
                    var_anoms = anom_ds[f'{eof_var}_anom']
                    var_anoms_hem = var_anoms.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)
                    var_anoms_ucomp = anom_ds[f'ucomp_anom']
                    var_anoms_hem_ucomp = var_anoms_ucomp.sel(lat=hemisphere_slice_dict[hemisphere]).sel(pfull=pfull_slice)

                    if time_frame == 'season':
                        var_anoms_hem = var_anoms_hem.where(eof_ds['time'].dt.month.isin(hemisphere_month_dict[hemisphere]), drop=True) 
                        time_dim_name = f'time_{hemisphere}'
                    else:
                        time_dim_name = 'time'

                    # Remove the time mean from the anomalies
                    var_anoms_hem = var_anoms_hem - var_anoms_hem.mean('time', skipna=False)
                    # Project the anomalies onto the ucomp EOFs to obtain pseudo PCs
                    pseudo_pcs = ucomp_solver.projectField(var_anoms_hem.values, neofs=n_eofs, eofscaling=1)
                    eof_ds[f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}'] = ((time_dim_name, 'eof_num'), pseudo_pcs)
                    eof_ds[f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}'] = eof_ds[f'{eof_var}_PCs_from_ucomp_{hemisphere}_{time_frame}'].transpose('eof_num', time_dim_name)
                    logging.info(f'made it for {eof_var} {hemisphere} {time_frame}')

        # Save the EOF dataset to file and then reopen it
        eof_ds.to_netcdf(output_eof_file_use)   
        eof_ds.close()
        eof_ds = xar.open_mfdataset(output_eof_file_use, decode_times=True)

    return eof_ds  

def arctan_omega(xdata_in, tau):
    """
    Compute the arctangent of (xdata_in * tau), used in curve fitting for phase difference.

    Parameters:
        xdata_in (numpy.ndarray): Input x data (typically angular frequency).
        tau (float): Fitted timescale parameter.

    Returns:
        numpy.ndarray: The computed arctan(xdata_in * tau) values.
    """
    return np.arctan(xdata_in * tau)

def power_spectrum_analysis(eof_ds, plot_dir, use_div1_proj, scaling_density=True):
    """
    Perform power spectrum analysis and plot the results using Welch and STFT methods.

    Parameters:
        eof_ds (xarray.Dataset): Dataset containing EOF-derived principal components.
        plot_dir (str): Directory where the plots will be saved.
        use_div1_proj (bool): Flag indicating whether to use projected div1 values.
        scaling_density (bool): If True, use 'density' scaling; otherwise, use 'spectrum'.

    Returns:
        None
    """
    # Create directory for power spectrum plots if it does not exist
    plot_dir_PS = f'{plot_dir}/power_spec_plots/'
    if not os.path.isdir(plot_dir_PS):
        os.makedirs(plot_dir_PS)

    # Initialize an empty dataset to hold power spectrum data
    power_spec_ds = xar.Dataset()

    # Loop over selected variables and hemispheres for analysis
    for var_to_analyse in ['div1_QG', 'div2_QG', 'fvbarstar']:
        for hemisphere in ['n', 's']:
            if scaling_density:
                scaling = 'density'
            else:
                scaling = 'spectrum'

            # Determine variable name based on whether projected div1 is used
            if use_div1_proj:
                div1_name = f'{var_to_analyse}_PCs_from_ucomp_{hemisphere}_all_time'
            else:
                div1_name = f'{var_to_analyse}_PCs_{hemisphere}_all_time'

            ucomp_name = f'ucomp_PCs_{hemisphere}_all_time'

            # Loop over the ucomp and div1 variables to compute power spectra
            for ps_var in [ucomp_name, div1_name]:
                # Extract the first principal component (assuming index 0) and remove extra dimensions
                ps_data = eof_ds[ps_var][0, ...].squeeze()
                # Remove the time mean from the data
                ps_data = ps_data - ps_data.mean('time').values
                ps_data_values = ps_data.values

                # Compute segmented power spectrum and update the dataset
                power_spec_ds = compute_segmented_power_spectrum(ps_data, scaling, power_spec_ds)
                plt.figure()
                # Plot the Welch-based power spectrum
                plt.plot(power_spec_ds['frequency'], power_spec_ds[f'{ps_var}_power_spec_welch'], label='welch')
                # Plot the STFT-based power spectrum (note factor 2 difference)
                plt.plot(power_spec_ds['frequency'], 2. * power_spec_ds[f'{ps_var}_power_spec_stft'], linestyle='--', label='stft')
                plt.legend()
                plt.xlim(0., 0.25)
                plt.title(f'Power spectrum of {ps_var} using Welch and STFT method')
                plt.xlabel('frequency (1/days)')
                plt.savefig(f'{plot_dir_PS}/{ps_var}_PC1_power_spectrum_welch_{scaling}.pdf')

            # Extract data for cospectral analysis
            cospec_data1 = eof_ds[ucomp_name][0, ...].squeeze()
            cospec_data2 = eof_ds[div1_name][0, ...].squeeze()

            # Compute cospectrum between ucomp and div1
            power_spec_ds = compute_cospectrum(cospec_data1, cospec_data2, power_spec_ds, ucomp_name=ucomp_name, div1_name=div1_name)
            # Compute phase difference based on cospectrum
            power_spec_ds = compute_phase_difference(power_spec_ds, div1_name, ucomp_name)

            # Plot the cospectrum (real and imaginary parts)
            plt.figure()
            plt.plot(power_spec_ds['frequency'], np.real(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_welch']), label='real Welch')
            plt.plot(power_spec_ds['frequency'], np.imag(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_welch']), label='Imag Welch')
            plt.plot(power_spec_ds['frequency'], np.real(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft']), label='real STFT', linestyle='--')
            plt.plot(power_spec_ds['frequency'], np.imag(power_spec_ds[f'{ucomp_name}_{div1_name}_cospec_stft']), label='Imag STFT', linestyle='--')    
            plt.plot(power_spec_ds['frequency'], 2. * np.pi * power_spec_ds['frequency'], label='2piomega')
            plt.xlim(0., 0.25)
            plt.legend()
            plt.title(f'Cospectrum of ucomp and div1')
            plt.xlabel('frequency (1/days)')
            plt.savefig(f'{plot_dir_PS}/ucomp_{div1_name}_PC1_cospectrum_{scaling}.pdf')    

            # Plot the squared coherence from Welch and STFT methods
            plt.figure()
            plt.plot(power_spec_ds['frequency'], power_spec_ds[f'{ucomp_name}_{div1_name}_coher_welch']**2., label='welch')
            plt.plot(power_spec_ds['frequency'], power_spec_ds[f'{ucomp_name}_{div1_name}_coher_stft']**2., label='stft', linestyle='--')
            plt.xlim(0., 0.25)
            plt.legend()
            plt.title(f'Coherence squared of ucomp and div1 using Welch method')
            plt.xlabel('frequency (1/days)')
            plt.savefig(f'{plot_dir_PS}/ucomp_{div1_name}_PC1_coher_sqd.pdf')      

            # Plot phase difference and fitted curves
            plt.figure()
            plt.plot(power_spec_ds['frequency'], power_spec_ds[f'{ucomp_name}_{div1_name}_phase_diff'], label='data')
            for i in range(1, 4):
                key = f"{div1_name}_{ucomp_name}_phase_diff_tau_fit_{i}"
                plt.plot(
                    power_spec_ds["frequency"],
                    np.rad2deg(np.arctan(2. * np.pi * power_spec_ds["frequency"] * power_spec_ds[key])),
                    linestyle="--",
                    label=f"fit with {power_spec_ds[key]:4.2f} days"
                )
            plt.xlim(0., 0.25)
            plt.legend()
            plt.title(f'Phase of ucomp and div1 using stft method')
            plt.xlabel('frequency (1/days)')
            plt.savefig(f'{plot_dir_PS}/ucomp_{div1_name}_PC1_phase_diff.pdf')    
