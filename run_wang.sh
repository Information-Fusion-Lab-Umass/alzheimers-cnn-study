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
--run_id=$SLURM_JOB_ID \
--log_level 20 \
--no_write_tensorboard \
--log_to_stdout \
--no_log_to_file \
--no_save_best_model \
--use_gpu \
--data_path data/NDJ_wang_splitByPaper_data.csv \
--brain_mask_path=$MASK_ICV_PATH \
--image_columns brain_mri_path \
--label_column DX \
--training_crossval_folds 10 \
--testing_split 0.2 \
--num_workers 0 \
--engine wang_densenet \
--pretrain_optim_lr 0.001 \
--pretrain_optim_wd 0.01 \
--pretrain_batch_size 2 \
--train_epochs 150 \
--train_optim_lr 0.001 \
--train_optim_wd 0.0005 \
--train-batch-size 10 \
--train_momentum 0.9 \
--validate_batch_size 10 \
--test_batch_size 10 \
--lrate_scheduler poly \
--optimizer SGD \
--image-column MRI_path \
--label-column DX \
--data-lookup /mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/wang_splitByMRI.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
