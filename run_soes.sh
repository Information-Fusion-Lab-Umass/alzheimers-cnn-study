#!/bin/bash
#
#SBATCH --job-name=debug-script
#SBATCH -e outputs/errors/%j.txt        # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=m40-long
#SBATCH --time=04-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=230000
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
--training_crossval_folds 10 \
--testing_split 0.2 \
--num_workers 0 \
--pretrain-optim-lr 0.001 \
--pretrain_optim_wd 0.01 \
--pretrain_batch_size 5 \
--pretrain-epochs 20 \
--train-epochs 150 \
--train-optim-lr 0.01 \
--train-optim-wd 0.0005 \
--train-batch-size 10 \
--train_momentum 0.9 \
--validate_batch_size 10 \
--test_batch_size 10 \
--lrate_scheduler poly \
--optimizer SGD \
--image-column MRI_path \
--label-column DX \
--engine soes_cnn
--data-lookup /mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/soes_splitByPaper.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
