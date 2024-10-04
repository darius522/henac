#!/bin/bash

#SBATCH -J dac
#SBATCH -p gpu
#SBATCH -o /home/daripete/jstsp-dac/logs/%j.out
#SBATCH -e /home/daripete/jstsp-dac/logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daripete@iu.edu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH -A r00105

conda activate /N/slate/daripete/anaconda3/envs/dac

CODE_ROOT="$PWD"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
                    --master-port 29401 \
                    --nproc_per_node gpu scripts/train.py \
                    --args.load conf/final/24khz_mb_32cb.yml \
                    --save_path runs/quant_1024_dropout=05_nocom_noadv/