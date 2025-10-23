import os
import json
import pandas as pd
from glob import glob

base_dir = "/home/links/ct715/eddy_feedback/chapter1/annual_cycle/data/PAMIP"   # <-- change this
out_dir = "/home/links/ct715/eddy_feedback/chapter1/zz_paper_plots/data" # <-- change this if you want outputs somewhere else
os.makedirs(out_dir, exist_ok=True)

# Hemisphere + season mapping
targets = {"nh": "DJF", "sh": "JAS"}

for hemi, season in targets.items():
    rows = []
    for json_file in glob(os.path.join(base_dir, "*", "efp_results.json")):
        model = os.path.basename(os.path.dirname(json_file))
        with open(json_file, "r") as f:
            data = json.load(f)

        # Grab only the requested season
        row = {
            "model": model,
            f"efp_{hemi}": data[f"efp_{hemi}"][season]["efp"],
            f"efp_{hemi}_gt3": data[f"efp_{hemi}_gt3"][season]["efp"],
            f"efp_{hemi}_123": data[f"efp_{hemi}_123"][season]["efp"],
        }
        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Save CSV per hemisphere
    out_file = os.path.join(out_dir, f"efp_{hemi}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")
