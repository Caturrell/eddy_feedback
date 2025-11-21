import os
import json
import pandas as pd

base_dir = '/home/links/ct715/eddy_feedback/chapter1/cmip6/data/30y'

rows = []

# Loop through model subdirectories
for model_dir in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_dir)
    if not os.path.isdir(model_path):
        continue  # skip if not a directory

    # find JSON file in this directory
    json_files = [f for f in os.listdir(model_path) if f.endswith('_efp_results_CMIP6_piControl_30y.json')]
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
csv_path = os.path.join(base_dir, "cmip6_efp_winters_30y.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Saved summary CSV to {csv_path}")
print(df)
