#!/bin/bash

#SBATCH -J resample_fma
#SBATCH -p general
#SBATCH -o /N/slate/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /N/slate/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --time=10:00:00
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/jamendo

CODE_ROOT="$PWD"

python resample_fma.py