#!/bin/bash

# Submit one training job from a copied model directory.
# Edit these variables after copying model_copy to e.g. 20x20_model.
SPAD_SIZE=${SPAD_SIZE:-20x20}
TENSOR_DIR=${TENSOR_DIR:-/lustre/work/$USER/SPAD_results/train_${SPAD_SIZE}}
MODEL_BASE=${MODEL_BASE:-NN_model}
GROUP=${GROUP:-}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
WORKERS=${WORKERS:-8}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
VAL_SPLIT=${VAL_SPLIT:-0.30}
EPOCH_SAMPLES=${EPOCH_SAMPLES:-0}
PARTITION=${PARTITION:-matador}
MEMORY=${MEMORY:-32G}
CPUS=${CPUS:-8}
TIME_SLICES=${TIME_SLICES:-}
PYTHON_ENV_BIN=${PYTHON_ENV_BIN:-$HOME/miniconda3/envs/dsipm-spad/bin}
if [ ! -x "$PYTHON_ENV_BIN/python3" ] && [ -x "$HOME/miniconda3/envs/base/bin/python3" ]; then
    PYTHON_ENV_BIN="$HOME/miniconda3/envs/base/bin"
fi

mkdir -p LOGDIR

gen_script() {
    local script_name="NN_Training_${SPAD_SIZE}.sh"
    local group_arg=""
    if [ -n "${GROUP}" ]; then
        group_arg="--group ${GROUP}"
    fi
    local epoch_samples_arg=""
    if [ "${EPOCH_SAMPLES}" -gt 0 ]; then
        epoch_samples_arg="--epoch-samples ${EPOCH_SAMPLES}"
    fi

    cat << EOF > "$script_name"
#!/bin/bash
#SBATCH -J "NN_train_${SPAD_SIZE}"
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
export PATH="${PYTHON_ENV_BIN}:\$PATH"
echo "Environment loaded."
export TIME_SLICES="${TIME_SLICES}"

python3 -u train.py \
  "${TENSOR_DIR}" \
  --spad "${SPAD_SIZE}" \
  --recursive \
  --base-dir "${MODEL_BASE}" \
  --epochs "${EPOCHS}" \
  --bs "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LEARNING_RATE}" \
  --val-split "${VAL_SPLIT}" \
  ${group_arg} \
  ${epoch_samples_arg}

echo "Training finished"
EOF

    chmod +x "$script_name"
    sbatch "$script_name"
}

gen_script
