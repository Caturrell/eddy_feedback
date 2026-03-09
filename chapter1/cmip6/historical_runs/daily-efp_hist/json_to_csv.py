import os
import json
import pandas as pd

YEAR_RANGE = (1958, 2014)

base_dir = f'/home/users/cturrell/documents/eddy_feedback/chapter1/cmip6/historical_runs/data/{YEAR_RANGE[0]}_{YEAR_RANGE[1]}/daily'

rows = []

# Loop through model subdirectories
for model_dir in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_dir)
    if not os.path.isdir(model_path):
        continue  # skip if not a directory

    # find JSON file in this directory
    json_files = [f for f in os.listdir(model_path) if f.endswith(f'_efp_CMIP6_historical_{YEAR_RANGE[0]}_{YEAR_RANGE[1]}.json')]
    if not json_files:
        print(f"⚠️ No JSON found in {model_path}")
        continue

    json_path = os.path.join(model_path, json_files[0])

    with open(json_path) as f:
        data = json.load(f)

    # helper to safely get value
    def get_val(section, season):
        try:
            return data[section][season]["efp"]
        except KeyError:
            return None

    # build one row
    row = {
        "model": model_dir,
        "efp_nh": get_val("efp_nh", "DJF"),
        "efp_nh_gt3": get_val("efp_nh_gt3", "DJF"),
        "efp_nh_123": get_val("efp_nh_123", "DJF"),
        "efp_sh": get_val("efp_sh", "JAS"),
        "efp_sh_gt3": get_val("efp_sh_gt3", "JAS"),
        "efp_sh_123": get_val("efp_sh_123", "JAS"),
    }

    rows.append(row)

# create dataframe
df = pd.DataFrame(rows)
df = df.sort_values("model").reset_index(drop=True)

# save CSV summary in base dir
csv_path = os.path.join(base_dir, '..', f"cmip6_hist_{YEAR_RANGE[0]}_{YEAR_RANGE[1]}_daily_efp_winters.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Saved summary CSV to {csv_path}")
print(df)
