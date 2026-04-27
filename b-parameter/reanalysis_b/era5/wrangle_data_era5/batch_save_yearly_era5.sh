#!/bin/bash
#SBATCH --job-name=batch_yearly_era5
#SBATCH --output=save_yearly_era5_v250.out
#SBATCH --error=save_yearly_era5_v250.err
#SBATCH --account=arctic_connect
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=50G            # reduced: chunked single-file processing needs far less
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1

# Activate your environment
mamba activate eddy

# Run your Python script
python /home/users/cturrell/documents/eddy_feedback/b-parameter/reanalysis_b/era5/wrangle_data_era5/save_era5_uv_yearly.py