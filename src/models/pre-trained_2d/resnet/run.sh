#!/bin/bash
#
#SBATCH --job-name=densenet
#SBATCH -e outputs/errors/%j.txt        # File to which STDERR will be written
#SBATCH --output=outputs/logs/%j.txt    # Output file
#SBATCH --partition=m40-long
#SBATCH --ntasks=12                     # Set to max_workers + 2
#SBATCH --time=04-00:00                 # Runtime in D-HH:MM
#SBATCH --mem=90000
#SBATCH --gres=gpu:2

python3 routine.py
sleep 1
exit
