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
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

# "/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_xlarge_fr_80/objective/input"
# "/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_xlarge_fr_80/objective/output_[24, 1]_18kbps"
# "/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_xlarge_fr_80/objective/output_[31, 1]_23kbps"

# "/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/objective/input"
# "/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/objective/output_[16, 4, 0]_18kbps"
# "/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/objective/output_[16, 4, 2]_23kbps"

python -u scripts/evaluate.py --input "/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/objective/input" --output "/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/objective/output_[16, 4, 2]_23kbps" --n_proc 8