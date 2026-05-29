#!/bin/bash

mkdir -p LOGDIR

gen_script() {
    local script_name="NN_Training.sh"

    cat << EOF > "$script_name"
#!/bin/bash
#SBATCH -J "NN_Training"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p matador
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

set -eou pipefail

# Load environment needed for python imports
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:\$PATH
echo "Environment loaded."

echo "Begin training"
python3 -u train.py \
  /lustre/work/samumcki/SPAD_results/vqe_test_200x200 \
  --spad 200x200 \
  --recursive \
  --bs 256 \
  --workers 8 \
  --predict-only \
  --pred-csv pred_vqe_test_200x200.csv

echo "Training finished"

EOF

    chmod +x "$script_name"
    sbatch "$script_name"

}

gen_script