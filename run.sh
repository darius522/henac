#!/bin/bash

#SBATCH -J bsline_large
#SBATCH -p hopper
#SBATCH -o /N/slate/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /N/slate/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
                    --nproc_per_node gpu scripts/train.py \
                    --args.load conf/final/32khz_baseline_small_2.yml \
                    --seed 1 \
                    --save_path runs_32khz/baseline_26cb_large_fr_80