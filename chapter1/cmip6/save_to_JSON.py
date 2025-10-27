import xarray as xr
import os
import json

path = '/gws/nopw/j04/arctic_connect/cturrell/CMIP6/piControl/efp_data_sit'

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

# open all datasets
ds = {}
for file in os.listdir(path):
    if file.endswith('.nc'):
        model_name = file.split('_')[0]
        ds[model_name] = xr.open_dataset(os.path.join(path, file))

# process each model
for model_name, model_ds in ds.items():
    result = {}

    for var in model_ds.data_vars:
        name_comps = var.split('_')

        # skip any "_time" variable
        if 'time' in name_comps[-1] or 'time' in name_comps[-2]:
            continue

        # extract season
        season_code = name_comps[-1]

        # extract hemisphere
        hemisphere = name_comps[-2]
        if hemisphere not in ['n', 's']:
            print(f"⚠️ Skipping unexpected var: {var}")
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

    # # compute "total_mon" = 123 + gt3
    # for hemi in ['nh', 'sh']:
    #     total_key = f"efp_{hemi}_total_mon"
    #     result[total_key] = {}

    #     for season in season_months.keys():
    #         efp_total = 0.0
    #         if f"efp_{hemi}_123" in result and season in result[f"efp_{hemi}_123"]:
    #             efp_total += result[f"efp_{hemi}_123"][season]["efp"]
    #         if f"efp_{hemi}_gt3" in result and season in result[f"efp_{hemi}_gt3"]:
    #             efp_total += result[f"efp_{hemi}_gt3"][season]["efp"]

    #         result[total_key][season] = {
    #             "efp": efp_total,
    #             "months": season_months[season]
    #         }

    # save JSON per model
    save_path = '/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/data'
    save_dir = os.path.join(save_path, f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    out_file = os.path.join(save_dir, f"{model_name}_efp_results_CMIP6_piControl.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Saved {out_file}")
