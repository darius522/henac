#!/bin/bash

#SBATCH -J dac
#SBATCH -p gpu
#SBATCH -o /N/slate/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /N/slate/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
                    --nproc_per_node gpu scripts/train.py \
                    --args.load conf/final/24khz.yml \
                    --save_path runs/scratch_baseline_24khz_lr_1e-4/ 