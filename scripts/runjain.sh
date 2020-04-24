#!/bin/bash
#
#SBATCH --job-name=jain-et-al
#SBATCH --partition=titanx-long
#SBATCH --time=04-00:00                 
#SBATCH --mem=240000
#SBATCH --gres=gpu:2
#SBATCH --ntasks=12 

#
export cmd="python main.py \
--method jain \
--epochs 50 \
--batch-size 40 \
--data-split-ratio 0.8 0.2 \
--data-path data/jain_splitByMRI.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
