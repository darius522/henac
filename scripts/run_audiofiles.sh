#!/bin/bash

#SBATCH -J eval
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH -o /N/slate/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /N/slate/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

# [31, 1]
# [27, 1]
# [25, 1]
# [22, 1]
# [20, 1]
# [18, 1]
python scripts/run_audiofiles.py --num_codebooks "[21,1]"