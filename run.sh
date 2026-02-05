#!/bin/bash

#SBATCH -J hb
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH -o /N/slate/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /N/slate/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
                    --nproc_per_node gpu scripts/train.py \
                    --args.load conf/final/24khz_hb.yml \
                    --seed 1 \
                    --save_path runs_24khz_bis/hb_18cb_2cb_fixed_1cb_fr_75_500_1000