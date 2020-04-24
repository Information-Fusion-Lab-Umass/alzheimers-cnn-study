#!/bin/bash
#
#SBATCH --job-name=liu-et-al
#SBATCH --partition=titanx-long
#SBATCH --time=04-00:00                 
#SBATCH --mem=240000
#SBATCH --gres=gpu:2
#SBATCH --ntasks=12 

#
export cmd="python main.py \
--method liu \
--epochs 30 \
--batch-size 4 \
--data-split-ratio 0.7 0.15 0.15 \
--data-path data/liu_splitByPT.csv"

echo ""
echo "Executing \"$cmd\""
echo ""

$cmd
