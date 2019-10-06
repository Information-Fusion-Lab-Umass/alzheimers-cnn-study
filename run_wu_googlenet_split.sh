#!/bin/bash
#
#SBATCH --job-name=debug-script
#SBATCH -e outputs/errors/%j.txt        # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=m40-long
#SBATCH --ntasks=12                     # Set to max_workers + 2
#SBATCH --time=04-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=90000
#SBATCH --gres=gpu:2

# To make a boolean option False, simply prefix with "no-"
export cmd="python3 main.py \
--run-id=$SLURM_JOB_ID \
--log-level 20 \
--no-write-tensorboard \
--log-to-stdout \
--no-log-to-file \
--no-save-best-model \
--use-gpu \
--image-column IMGPATH \
--label-column DX \
--dataset-size-limit 10 \
--training-crossval-folds 5 \
--testing-split 0.2 \
--num-workers 0 \
--engine wu_googlenet \
--train-epochs 24 \
--train-optim-lr 0.01 \
--train-optim-wd 0.0005 \
--train-batch-size 10 \
--train-momentum 0.9 \
--validate-batch-size 10 \
--test-batch-size 10 \
--optimizer SGD"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
