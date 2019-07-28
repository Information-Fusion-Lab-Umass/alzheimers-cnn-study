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
--use_gpu \
--data_path data/NDJ_soes_splitByPaper_data.csv \
--brain_mask_path=$MASK_ICV_PATH \
--image_columns brain_mri_path \
--label_column DX \
--num_classes 2
--training_crossval_folds 7
--testing_split 0.05 \
--num_workers 0 \
--engine soes_3d \
--pretrain_optim_lr 0.001 \
--pretrain_optim_wd 0.01 \
--pretrain_batch_size 2 \
--train_epochs 300 \
--train_optim_lr 0.000001 \
--train_optim_wd 0.5 \
--train_batch_size 10 \
--validate_batch_size 10 \
--test_batch_size 10 \
--optimizer Adam"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
