import os
import logging
import xarray as xr
import functions.data_wrangling as dw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MAIN_PATH = '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/psl'
OUTPUT_PATH = '/home/links/ct715/data_storage/PAMIP/processed_monthly/mslp'

failed_files = []

def process_hadgem3(files, model_path, model):
    for i in range(0, len(files), 2):
        file_pair = files[i:i+2]
        if len(file_pair) < 2:
            logging.warning(f"Incomplete file pair for {model}: {file_pair}")
            failed_files.append((model, file_pair))
            continue

        file_root = '_'.join(file_pair[0].split('_')[:-1])
        output_file = os.path.join(OUTPUT_PATH, model, f'{file_root}_200006-200105_processed.nc')

        if os.path.exists(output_file):
            logging.info(f"Output already exists: {os.path.basename(output_file)}")
            continue

        file_paths = [os.path.join(model_path, f) for f in file_pair]
        try:
            ds = xr.open_mfdataset(file_paths, combine='by_coords')
            ds = dw.data_checker1000(ds)
            ds = ds.sel(time=slice('2000-06', '2001-05'))
            ds.to_netcdf(output_file)
            logging.info(f"Processed and saved: {output_file}")
        except Exception as e:
            logging.error(f"Error processing files {file_pair} for model {model}: {e}")
            failed_files.append((model, file_pair))

def process_standard_model(files, model_path, model):
    for file in files:
        if not file.endswith('.nc'):
            continue

        file_root = '_'.join(file.split('_')[:-1])
        output_file = os.path.join(OUTPUT_PATH, model, f'{file_root}_200006-200105_processed.nc')

        if os.path.exists(output_file):
            logging.info(f"Output already exists: {os.path.basename(output_file)}")
            continue

        file_path = os.path.join(model_path, file)
        try:
            ds = xr.open_dataset(file_path)
            ds = dw.data_checker1000(ds)
            ds = ds.sel(time=slice('2000-06', '2001-05'))

            if 'p' in ds.data_vars:
                ds = ds.rename({'p': 'psl'})
                ds = ds.isel(msl=0)
            if 'msl' in ds.data_vars:
                ds = ds[['psl']]

            ds.to_netcdf(output_file)
            logging.info(f"Processed and saved: {output_file}")
        except Exception as e:
            logging.error(f"Error processing file {file} for model {model}: {e}")
            failed_files.append((model, file))

def main():
    models = sorted(os.listdir(MAIN_PATH))

    for model in models:
        model_path = os.path.join(MAIN_PATH, model)
        if not os.path.isdir(model_path):
            continue

        logging.info(f'Processing model: {model}')
        files = sorted(os.listdir(model_path))

        if model == 'HadGEM3-GC31-MM':
            process_hadgem3(files, model_path, model)
        else:
            process_standard_model(files, model_path, model)

    # Summary of failures
    if failed_files:
        logging.warning("\n--- The following files failed to process ---")
        for entry in failed_files:
            logging.warning(f"Model: {entry[0]}, File(s): {entry[1]}")
    else:
        logging.info("All files processed successfully!")

if __name__ == '__main__':
    main()
