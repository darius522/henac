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
#SBATCH --time=05:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

python scripts/evaluate.py --input "/N/slate/daripete/jstsp-dac/runs2/hb_18cb/300k/audios/input" --output "/N/slate/daripete/jstsp-dac/runs2/hb_18cb/300k/audios/output"