import xarray as xr
import pandas as pd
import os

def split_netcdf_by_year(input_path, output_dir=None, time_dim='time'):
    """
    Splits a NetCDF file into separate yearly datasets.
    
    Parameters:
        input_path (str): Path to the NetCDF file.
        output_dir (str, optional): Directory to save yearly NetCDFs. If None, does not save.
        time_dim (str): Name of the time dimension in the NetCDF file.
    
    Returns:
        dict: A dictionary mapping year -> xarray.Dataset
    """
    ds = xr.open_dataset(input_path)
    
    if time_dim not in ds.coords and time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset.")
    
    # Make sure time is datetime-like
    time_index = pd.to_datetime(ds[time_dim].values)
    
    # Add year as a coordinate
    ds = ds.assign_coords(year=("time", time_index.year))
    
    # Group by year
    yearly_datasets = {}
    
    for year, ds_year in ds.groupby('year'):
        yearly_datasets[year] = ds_year
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{year}_daily_averages.nc")
            ds_year.to_netcdf(output_path)
            print(f"Saved: {output_path}")
    
    return yearly_datasets

if __name__ == "__main__":
    path = '/home/links/ct715/data_storage/reanalysis/jra55_daily'
    input_file = os.path.join(path, "jra55_uvtw_ubar_ep-QG_k.nc")      # <-- replace with your file
    output_folder = os.path.join(path, "k123_QG_epfluxes")   # <-- or set to None if you don’t want to save

    yearly_data = split_netcdf_by_year(input_file, output_dir=output_folder)
    print(f"Split into {len(yearly_data)} years: {list(yearly_data.keys())}")
