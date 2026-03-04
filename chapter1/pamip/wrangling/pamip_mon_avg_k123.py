import xarray as xr
import os

def process_directory(directory, output_base_name):
    """
    Process all NetCDF files in each model directory, combine ensemble members,
    and save each model's data into a single NetCDF file.

    Args:
        directory (str): Base directory containing model subdirectories.
        output_base_name (str): Base name for output files.
    """

    models_list = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for model in models_list:
        model_dir = os.path.join(directory, model)
        files = [f for f in os.listdir(model_dir) if f.endswith('.nc')]
        files.sort()
        num_ens_mems = len(files)

        if num_ens_mems == 0:
            print(f"No NetCDF files found for model: {model}")
            continue

        print(f"Processing model: {model} with {num_ens_mems} ensemble members.")

        try:
            # Open all ensemble member files for the current model
            ds = xr.open_mfdataset([os.path.join(model_dir, f) for f in files], combine='nested', concat_dim='ens_ax')

            # Define output filename and path
            output_filename = f'{model}_{output_base_name}_r{num_ens_mems}.nc'
            output_path = os.path.join(directory, output_filename)

            # Save the combined dataset
            ds.to_netcdf(output_path)
            print(f"Combined ensemble members saved to {output_path}")

        except Exception as e:
            print(f"Error processing model {model}: {e}")

# Example usage
print("Processing data...")


base_dir = '/home/links/ct715/data_storage/PAMIP/processed_daily/k123_daily_efp_mon-avg'

process_directory(base_dir, 'k123_mm')
