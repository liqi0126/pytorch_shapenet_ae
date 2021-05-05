#!/bin/bash
#SBATCH --partition=orion --qos=normal
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:titanxp:1

#SBATCH --job-name="vae1e-5"
#SBATCH --output="scripts/train_v3_fcupconv_emd_vae-1e-5.out"

# only use the following if you want email notification
####SBATCH --mail-user=daerduomkch@gmail.com
####SBATCH --mail-type=ALL

python ./train_v1.py  \
    --exp_suffix fcupconv-emd-vae-1e-5 \
    --data_dir ../../data/PartNetShapeData-Chair/train \
    --val_data_dir ../../data/PartNetShapeData-Chair/val \
    --num_point 2048 \
    --decoder_type fcupconv \
    --loss_type emd \
    --probabilistic \
    --kldiv_loss_weight 1e-5 \
    --model_version model_v3 \
    --overwrite
