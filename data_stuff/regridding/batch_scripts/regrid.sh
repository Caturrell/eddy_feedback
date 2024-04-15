#!/bin/bash

#SBATCH --partition=short-serial
#SBATCH --job-name=pamip_regrid
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=12:00:00

module add jaspy