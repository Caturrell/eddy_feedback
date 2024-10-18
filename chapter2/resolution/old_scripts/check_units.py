import xarray as xr
import os
import pdb

# extract model names from directory
path = '/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/epfy/'
model_names = os.listdir(path)
model_names.remove('OpenIFS-1279')
model_names.remove('OpenIFS-159')
model_names.remove('OpenIFS-511')

# loop over each model to check units for the variables and ensure no corrupt files
for item in model_names:
    print(f'Checking {item}:')
    
    for variable in ['ua', 'epfy']:
        
        # define path to both sets of data for each model
        var_path = f'/home/links/ct715/data_storage/PAMIP/monthly/1.1_pdSST-pdSIC/{variable}/'
        model_data_path = os.path.join(var_path, item, '*.nc')
        
        # open dataset
        ds = xr.open_mfdataset(model_data_path, cache=False, parallel=True, chunks={'time':31},
                               concat_dim='ens_ax', combine='nested')
        
        # pdb.set_trace()
        
        # check units
        try:
            units = ds[variable].attrs.get('units', 'No units specified')
        except KeyError as e1:
            print(f'{e1}')
            units = f'Name not {variable}'
            print(f'ISSUE: {item} - {variable}')
        
        print(f'{item}: {variable}: {units}')