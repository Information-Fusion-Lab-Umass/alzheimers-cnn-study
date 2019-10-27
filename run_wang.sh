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
--training-crossval-folds 10 \
--num-workers 6 \
--engine wang_densenet \
--train-epochs 15 \
--train-optim-lr 0.01 \
--train-optim-wd 0.0005 \
--train-batch-size 10 \
--train-momentum 0.9 \
--validate-batch-size 10 \
--test-batch-size 10 \
--lrate-scheduler poly \
--train-optimizer SGD \
--image-column MRI_path \
--label-column DX \
--data-lookup /mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/wang_splitByPaper.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
