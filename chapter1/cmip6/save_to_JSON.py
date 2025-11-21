import xarray as xr
import os
import json

path = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit/30y'

# month mapping for each season code
season_months = {
    "DJF": [12, 1, 2],
    "JFM": [1, 2, 3],
    "FMA": [2, 3, 4],
    "MAM": [3, 4, 5],
    "AMJ": [4, 5, 6],
    "MJJ": [5, 6, 7],
    "JJA": [6, 7, 8],
    "JAS": [7, 8, 9],
    "ASO": [8, 9, 10],
    "SON": [9, 10, 11],
    "OND": [10, 11, 12],
    "NDJ": [11, 12, 1]
}

print("="*60)
print("Starting EFP data processing")
print("="*60)
print(f"Input path: {path}")

# open all datasets
ds = {}
nc_files = [f for f in os.listdir(path) if f.endswith('.nc')]
print(f"\n📂 Found {len(nc_files)} NetCDF files")

for file in nc_files:
    model_name = file.split('_')[0]
    print(f"  Loading: {file} (model: {model_name})")
    ds[model_name] = xr.open_dataset(os.path.join(path, file))

print(f"\n✅ Successfully loaded {len(ds)} models")
print("="*60)

# process each model
for i, (model_name, model_ds) in enumerate(ds.items(), 1):
    print(f"\n[{i}/{len(ds)}] Processing model: {model_name}")
    print(f"  Variables found: {len(model_ds.data_vars)}")
    
    result = {}
    vars_processed = 0
    vars_skipped = 0
    
    for var in model_ds.data_vars:
        name_comps = var.split('_')
        
        # skip any "_time" variable
        if 'time' in name_comps[-1] or 'time' in name_comps[-2]:
            vars_skipped += 1
            continue
        
        # extract season
        season_code = name_comps[-1]
        
        # extract hemisphere
        hemisphere = name_comps[-2]
        if hemisphere not in ['n', 's']:
            print(f"  ⚠️  Skipping unexpected var: {var}")
            vars_skipped += 1
            continue
        
        # determine wavenumber category
        if name_comps[-3] in ['123', 'gt3']:
            wavenumber = name_comps[-3]
            key = f"efp_{'nh' if hemisphere == 'n' else 'sh'}_{wavenumber}"
        else:
            wavenumber = 'total'
            key = f"efp_{'nh' if hemisphere == 'n' else 'sh'}"
        
        # get efp value (convert to float)
        efp_value = float(model_ds[var].values)
        
        # ensure key exists
        if key not in result:
            result[key] = {}
        
        # store data
        result[key][season_code] = {
            "efp": efp_value,
            "months": season_months[season_code]
        }
        vars_processed += 1
    
    print(f"  Processed: {vars_processed} variables")
    print(f"  Skipped: {vars_skipped} variables")
    print(f"  Result keys: {list(result.keys())}")
    
    # save JSON per model
    save_path = '/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/data/30y'
    save_dir = os.path.join(save_path, f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    out_file = os.path.join(save_dir, f"{model_name}_efp_results_CMIP6_piControl_30y.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"  ✅ Saved: {out_file}")

print("\n" + "="*60)
print(f"🎉 Processing complete! Processed {len(ds)} models")
print("="*60)