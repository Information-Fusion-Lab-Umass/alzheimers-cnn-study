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
export cmd="python main.py \
--run_id=$SLURM_JOB_ID \
--log_level 20 \
--no_write_tensorboard \
--log_to_stdout \
--log_to_file \
--no_save_best_model \
--use_gpu \
--label_column DX \
--num_classes 3 \
--training_crossval_folds 1 \
--testing_split 0.2 \
--num_workers 6 \
--train_epochs 50 \
--train_optim_lr 0.0001 \
--train_optim_wd 0.5 \
--train_batch_size 40 \
--validate_batch_size 40 \
--test_batch_size 40 \
--optimizer RMSprop \
--engine jain_vgg \
--image-column MRI_path \
--data-lookup /mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/jain_splitByMRI.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
