#!/bin/bash

# Submit one prediction job from a copied model directory.
# Usually run after trainNN.sh has produced NN_model_<SPAD_SIZE>/best_latest.ckpt.
SPAD_SIZE=${SPAD_SIZE:-20x20}
TENSOR_DIR=${TENSOR_DIR:-/lustre/work/$USER/SPAD_results/predict_${SPAD_SIZE}}
MODEL_BASE=${MODEL_BASE:-NN_model}
CHECKPOINT=${CHECKPOINT:-${PWD}/${MODEL_BASE}_${SPAD_SIZE}/best_latest.ckpt}
PRED_CSV=${PRED_CSV:-predictions_${SPAD_SIZE}.csv}
BATCH_SIZE=${BATCH_SIZE:-256}
WORKERS=${WORKERS:-8}
PARTITION=${PARTITION:-matador}
MEMORY=${MEMORY:-32G}
CPUS=${CPUS:-8}
TIME_SLICES=${TIME_SLICES:-}

mkdir -p LOGDIR

gen_script() {
    local script_name="NN_Predict_${SPAD_SIZE}.sh"

    cat << EOF > "$script_name"
#!/bin/bash
#SBATCH -J "NN_pred_${SPAD_SIZE}"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p ${PARTITION}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --gpus-per-node=1

set -euo pipefail

echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:\$PATH
echo "Environment loaded."
export TIME_SLICES="${TIME_SLICES}"

python3 -u train.py \
  "${TENSOR_DIR}" \
  --spad "${SPAD_SIZE}" \
  --recursive \
  --base-dir "${MODEL_BASE}" \
  --predict-only \
  --checkpoint "${CHECKPOINT}" \
  --bs "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --pred-csv "${PRED_CSV}"

echo "Prediction finished"
EOF

    chmod +x "$script_name"
    sbatch "$script_name"
}

gen_script
