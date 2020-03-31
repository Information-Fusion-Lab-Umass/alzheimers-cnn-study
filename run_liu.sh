#!/bin/bash
#
#SBATCH --job-name=wang-et-al
#SBATCH -e outputs/errors/%j.txt        # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=m40-long
#SBATCH --time=04-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=240000
#SBATCH --gres=gpu:2
#SBATCH --ntasks=12 

# To make a boolean option False, simply prefix with "no-"
export cmd="python main.py \
--run-id=$SLURM_JOB_ID \
--log-level 20 \
--no-write-tensorboard \
--log-to-stdout \
--no-log-to-file \
--no-save-best-model \
--use-gpu \
--testing-split 0.2 \
--num-workers 6 \
--engine liu \
--train-epochs 12 \
--train-optim-lr 0.01 \
--train-optim-wd 0.0 \
--train-batch-size 4 \
--train-momentum 0.9 \
--validate-batch-size 4 \
--test-batch-size 4 \
--train-optimizer SGD \
--image-column MRI_path \
--label-column DX \
--data-lookup /mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/liu_splitByPT.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
