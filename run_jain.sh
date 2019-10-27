#!/bin/bash
#
#SBATCH --job-name=jain-et-al
#SBATCH -e outputs/errors/%j.txt        # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=m40-long
#SBATCH --ntasks=12                     # Set to max_workers + 2
#SBATCH --time=04-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=90000
#SBATCH --gres=gpu:2


# To make a boolean option False, simply prefix with "no-"
export cmd="python main.py \
--run-id=$SLURM_JOB_ID \
--log-level 20 \
--no-write-tensorboard \
--log-to-stdout \
--no-log-to-file \
--no-save-best-model \
--use-gpu \
--label-column DX \
--num-classes 3 \
--training-crossval-folds 1 \
--testing-split 0.2 \
--num-workers 6 \
--train-epochs 50 \
--train-optim-lr 0.0001 \
--train-optim-wd 0.5 \
--train-batch-size 40 \
--validate-batch-size 40 \
--test-batch-size 40 \
--optimizer RMSprop \
--engine jain_vgg \
--image-column MRI_path \
--data-lookup /mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/jain_splitByMRI.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
