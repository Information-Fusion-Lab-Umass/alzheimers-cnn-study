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
--run_id=$SLURM_JOB_ID \
--log_level 20 \
--no_write_tensorboard \
--log_to_stdout \
--no_log_to_file \
--no_save_best_model \
--no_use_gpu \
--data_path data/BET_data_mapping_final.pickle \
--brain_mask_path=$MASK_ICV_PATH \
--image_columns brain_mri_path \
--label_column DX \
--training_crossval_folds 5
--testing_split 0.2 \
--num_workers 0 \
--engine wu_caffe_2d \
--pretrain_optim_lr 0.001 \
--pretrain_optim_wd 0.01 \
--pretrain_batch_size 2 \
--train_epochs 24 \
--train_optim_lr 0.01 \
--train_optim_wd 0.0005 \
--train_batch_size 10 \
--train_momentum 0.9 \
--validate_batch_size 10 \
--test_batch_size 10 \
--lrate_scheduler poly \
--optimizer SGD"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
