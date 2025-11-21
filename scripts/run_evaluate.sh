#!/bin/bash

#SBATCH -J eval
#SBATCH -p gpu
#SBATCH -o /N/slate/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /N/slate/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

# output_[16, 1, 0]_14kbps
# output_[16, 2, 0]_16kbps
# output_[16, 3, 0]_17kbps
# output_[16, 4, 0]_19kbps
# output_[16, 4, 1]_21kbps
# output_[16, 4, 2]_24kbps
python -u scripts/evaluate.py --input "/N/slate/daripete/jstsp-dac/runs_32khz/hb_16cb_4_1cb_2_1cb_fr_80_320_500/300k/audios/input" --output "/N/slate/daripete/jstsp-dac/runs_32khz/hb_16cb_4_1cb_2_1cb_fr_80_320_500/300k/audios/output_[16, 4, 2]_24kbps" --n_proc 1