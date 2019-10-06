#!/bin/bash
#
#SBATCH --job-name=wu-et-al
#SBATCH -e outputs/errors/%j.txt        # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=titanx-long
#SBATCH --ntasks=12                     # Set to max_workers + 2
#SBATCH --time=04-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=45000
#SBATCH --gres=gpu:1

# To make a boolean option False, simply prefix with "no-"
export cmd="python3 main.py \
--run-id=$SLURM_JOB_ID \
--log-level 20 \
--write-tensorboard \
--log-to-stdout \
--no-log-to-file \
--save-best-model \
--save-results \
--use-gpu \
--image-column IMGPATH \
--label-column DX \
--dataset-size-limit -1 \
--training-crossval-folds 5 \
--num-workers 6 \
--engine wu_googlenet \
--train-epochs 400 \
--train-batch-size 128 \
--validate-batch-size 256 \
--test-batch-size 256"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
