#!/bin/bash
#SBATCH -J "batch_npy"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 1
#SBATCH --mem-per-cpu=32G

#
#   Sum all npz's in a folder to a single .npy tensor
#   Used to make cummulative time-sliced output for reference
#

python3 -u sum_all_npz_to_npy.py \
/lustre/work/samumcki/SPAD_results/M_all_20x20 \ 
-o M_summed_20x20.npy \
--progress-every 200
