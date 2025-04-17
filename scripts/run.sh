#!/bin/bash

#SBATCH -J eval
#SBATCH -p general
#SBATCH -o /N/slate/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /N/slate/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --cpus-per-task=92
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

python scripts/evaluate.py