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
--save_result \
--log_to_stdout \
--no_log_to_file \
--no_save_best_model \
--use_gpu \
--data_path data/data_mapping.pickle \
--brain_mask_path=$MASK_ICV_PATH \
--image_columns skull_intact_path \
--label_column DX \
--training_crossval_folds 5
--testing_split 0.2 \
--num_workers 0 \
--engine resnet_3d \
--pretrain_optim_lr 0.001 \
--pretrain_optim_wd 0.01 \
--pretrain_batch_size 2 \
<<<<<<< HEAD
--train_epochs 1 \
--train_optim_lr 0.01 \
--train_optim_wd 0.0005 \
--train_batch_size 10 \
--train_momentum 0.9 \
--validate_batch_size 10 \
--test_batch_size 10 \
--lrate_scheduler poly \
--optimizer SGD"
=======
--train_epochs 10 \
--train_optim_lr 0.001 \
--train_optim_wd 0.01 \
--train_batch_size 6 \
--validate_batch_size 6 \
--test_batch_size 6 \
--optimizer Adam"
>>>>>>> ead5bae006c6404bfa76634c1dce272157323305

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
