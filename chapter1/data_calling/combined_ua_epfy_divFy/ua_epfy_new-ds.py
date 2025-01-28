import xarray as xr
from pathlib import Path
import logging
import pdb

import functions.data_wrangling as data
import functions.eddy_feedback as ef

import warnings
warnings.filterwarnings("ignore", category=xr.SerializationWarning)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# extract list of working models
model_list_path = Path('/home/links/ct715/data_storage/PAMIP/processed_monthly/regridded_3x3/1.6_pdSST-futArcSIC')
list_dir = model_list_path.iterdir()
model_list = [item.name.split('_')[0] for item in list_dir]
model_list.sort()
model_list.remove('OpenIFS-511')    # data not ready yet
model_list.remove('CESM1-WACCM-SC') # ua is zonal mean


#---------------------------------------------------------------------------------------------------------------------

data_dir = Path('/home/links/ct715/data_storage/PAMIP/monthly')
exp_list = ['1.1_pdSST-pdSIC', '1.6_pdSST-futArcSIC']
                
bad_files = []

for model in model_list:
    logging.info(f'Processing: {model}')
    for exp in exp_list:
        # Set save directory, create if it doesn't exist
        save_dir = data_dir / exp / 'combined_ua_epfy_divFy' / model
        save_dir.mkdir(exist_ok=True, parents=True)

        # Expand file lists for the model and experiment
        ua_files = sorted((data_dir / exp / 'ua' / model).glob('*.nc'))
        epfy_files = sorted((data_dir / exp / 'epfy' / model).glob('*.nc'))
        if len(ua_files) != len(epfy_files):
            bad_files.append(f'not equal ens members: {model}\n')
            continue
        else:
            ens_members = len(ua_files)

            if not ua_files or not epfy_files:
                logging.warning(f"No files found for model {model} and experiment {exp}. Skipping...")
                continue
            
            # pdb.set_trace()


            for i in range(ens_members):
                try:
                    # Open and combine datasets
                    ua = xr.open_mfdataset(
                        str(ua_files[i]),
                        parallel=True,
                        chunks={'time': 31},
                        combine='nested',
                        concat_dim='ens_ax'
                    )
                    epfy = xr.open_mfdataset(
                        str(epfy_files[i]),
                        parallel=True,
                        chunks={'time': 31},
                        combine='nested',
                        concat_dim='ens_ax'
                    )
                    
                    # match pressure levels to smaller dataset
                    if len(epfy.level) > len(ua.level):
                        epfy = epfy.sel( level = ua.level.values )
                    else:
                        ua = ua.sel( level = epfy.level.values )
                        
                    
                    # Save the processed dataset
                    if ens_members == 1:
                        output_file = save_dir / f'{model}_{exp.split("_")[0]}_rALL_ua_epfy_divFy.nc'
                    else:
                        output_file = save_dir / f'{model}_{exp.split("_")[0]}_r{i+1}_ua_epfy_divFy.nc'
                        
                    # Check if the file already exists
                    if output_file.exists():
                        # logging.info(f"File {output_file} already exists. Skipping...")
                        continue
                    else:
                        # Merge datasets and calculate divFy
                        ds = xr.Dataset({'ua': ua.ua, 'epfy': epfy.epfy})
                        ds = ef.calculate_divFphi(ds)
                        # Save the processed dataset
                        ds.to_netcdf(output_file)
                        logging.info(f"Saved processed dataset to {output_file}")

                except Exception as e:
                    logging.error(f"Error processing {model} in {exp}: {e}")
                    bad_files.append(f'Exception for model:{model} - experiment:{exp}. Error: {e}\n')

# Save bad_files to a .txt file
bad_files_path = Path('/home/links/ct715/eddy_feedback/chapter1/data_calling/combined_ua_epfy_divFy/bad_files.txt')
with bad_files_path.open('w') as f:
    f.writelines(bad_files)

logging.info(f"Saved list of bad files to {bad_files_path}")