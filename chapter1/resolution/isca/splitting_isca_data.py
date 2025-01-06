import xarray as xr
from pathlib import Path
import numpy as np

import functions.eddy_feedback as ef

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def seasonal_mean_datasets(ds):
    """
    Calculate seasonal means for a dataset, focusing on specific variables.

    Parameters:
        ds (xarray.Dataset): Input dataset containing time-series data.

    Returns:
        xarray.Dataset: Dataset resampled to seasonal means (DJF starting quarters)
                        and containing only the 'ubar' and 'divFy' variables.
    """
    ds = ds.sel(time=slice('2000-12', '2099-11'))
    ds = ds.resample(time='QS-DEC').mean('time')
    ds = ds[['ubar', 'divFy']]
    return ds

def calculate_efp_isca(ds, hemisphere):
    """
    Calculate the Eddy Feedback Parameter (EFP) using Isca data.

    Parameters:
        ds (xarray.Dataset): Dataset containing 'divFy' and 'ubar' variables.
        hemisphere (str): Hemisphere to analyze ('NH' for Northern Hemisphere, 'SH' for Southern Hemisphere).

    Returns:
        float: Eddy Feedback Parameter, rounded to 4 decimal places.

    Notes:
        - The function assumes the dataset contains a 'level' dimension for pressure levels
          and a 'lat' dimension for latitude.
        - The EFP is computed as the latitude-weighted mean of the squared correlation
          between 'divFy' and 'ubar', averaged over the specified hemisphere and pressure levels.
    """
    # Calculate the squared correlation between 'divFy' and 'ubar'
    correlation = xr.corr(ds.divFy, ds.ubar, dim='time')
    corr = correlation**2

    # Subset for EFP box based on pressure levels
    corr = corr.sel(level=slice(600., 200.))

    # Subset for the specified hemisphere
    if hemisphere == 'NH':
        corr = corr.sel(lat=slice(25, 75))
    elif hemisphere == 'SH':
        corr = corr.sel(lat=slice(-75, -25))
    else:
        raise ValueError("Hemisphere not specified. Use 'NH' or 'SH'.")

    # Average over pressure levels
    corr = corr.mean('level')

    # Apply latitude weighting and compute the mean
    weights = np.cos(np.deg2rad(corr.lat))
    eddy_feedback_param = corr.weighted(weights).mean('lat')

    return eddy_feedback_param.values.round(4)


# split data up
def split_and_process_data(ds, split_config):
    """
    Splits the dataset into subsets based on season, time range, and hemisphere,
    then calculates the EFP for each subset.

    Parameters:
    - ds (xarray.Dataset): The input dataset to be split.
    - calculate_efp_isca (function): Function to calculate EFP from a dataset.
    - split_config (dict): Dictionary defining the time splits, e.g.,
      {'set1': slice('2000-12', '2049-09'), 'set2': slice('2049-12', '2098-09')}.
    - hemispheres (dict): Dictionary defining hemispheric slices, e.g.,
      {'SH': slice(-90, 0), 'NH': slice(0, 90)}.
    - seasonal_months (dict): Dictionary defining seasonal months, e.g.,
      {'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11]}.

    Returns:
    - list: List of calculated EFP values for each subset.
    """
    
    seasonal_months = {
    'djf': 12,
    'mam': 3,
    'jja': 6,
    'son': 9
    }

    hemispheres = {'SH': slice(-90, 0), 'NH': slice(0,90)}
    
    subsets = {}
    efp_results = []

    # Iterate through seasonal months, splits, and hemispheres
    for season, months in seasonal_months.items():
        for split, split_range in split_config.items():
            for hemisphere, lat_range in hemispheres.items():

                # Subset dataset for the season (handling multi-month selection)
                season_ds = ds.sel(time=ds.time.dt.month.isin(months))

                # Subset data for the time range split
                split_ds = season_ds.sel(time=split_range)

                # Subset data for the hemisphere
                hemis_ds = split_ds.sel(lat=lat_range)

                # Store the subset
                key = f'{season}_{hemisphere}_{split}'
                subsets[key] = hemis_ds
                
                efp = calculate_efp_isca(hemis_ds, hemisphere=hemisphere)

                # # Calculate EFP for the subset
                # if hemisphere == 'NH':
                #     efp = ef.calculate_efp(hemis_ds, data_type='isca', calc_south_hemis=False)
                # elif hemisphere == 'SH':
                #     efp = ef.calculate_efp(hemis_ds, data_type='isca', calc_south_hemis=True)
                # else:
                #     raise ValueError(f'Correct hemisphere not specified: {hemisphere}')
                efp_results.append(efp)

    return efp_results

# Function to perform bootstrapping
def bootstrap(dataa, n_iterations=10000):
    
    n_size=len(dataa)   # calc size of data
    statistics = []     # To store bootstrap sample statistics
    
    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(dataa, size=n_size, replace=True)
        # Calculate statistic (e.g., mean) for this sample
        stat = np.mean(sample)
        statistics.append(stat)
        
    return statistics

def print_bootstrap_stats(dataa):
    
    # Perform bootstrapping
    bootstrap_means = bootstrap(dataa)
    # Calculate confidence interval (e.g., 95%)
    conf_interval = np.percentile(bootstrap_means, [2.5, 97.5])

    # Results
    print(f"Original Mean: {np.mean(dataa):.2f}")
    print(f"Bootstrap Mean: {np.mean(bootstrap_means):.2f}")
    print(f"95% Confidence Interval: {conf_interval}")
    
    