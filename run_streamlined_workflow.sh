#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p nocona
#SBATCH --mem=16G

# End-to-end dSiPM SPAD workflow runner.
# Run interactively or submit the controller with:
#   sbatch run_streamlined_workflow.sh
# Add cluster options on the command line if needed, e.g.
#   sbatch -p nocona --mem=16G run_streamlined_workflow.sh
# Override workflow variables at submission time, e.g.
#   sbatch --export=ALL,NN_EPOCH_SAMPLES=20000,MAX_JOB_RETRIES=3 run_streamlined_workflow.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_DIR=$SLURM_SUBMIT_DIR
else
    REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
cd "$REPO_DIR"

# ---------------- Editable workflow switches ----------------
RUN_GEANT4=${RUN_GEANT4:-1}
RUN_SPAD_TENSORS=${RUN_SPAD_TENSORS:-1}
RUN_PHOTON_FULL_ANALYSIS=${RUN_PHOTON_FULL_ANALYSIS:-1}
RUN_NN_TRAIN=${RUN_NN_TRAIN:-1}
RUN_NN_PREDICT=${RUN_NN_PREDICT:-1}
RUN_CUMULATIVE_NPY=${RUN_CUMULATIVE_NPY:-1}
RUN_PLOTS=${RUN_PLOTS:-1}
WAIT_FOR_JOBS=${WAIT_FOR_JOBS:-1}
MAX_JOB_RETRIES=${MAX_JOB_RETRIES:-2}
SLURM_POLL_SECONDS=${SLURM_POLL_SECONDS:-60}
SLURM_SETTLE_SECONDS=${SLURM_SETTLE_SECONDS:-15}
SLURM_ID_CHUNK_SIZE=${SLURM_ID_CHUNK_SIZE:-100}
MIN_ROOT_SIZE_MB=${MIN_ROOT_SIZE_MB:-1}
ROOT_READABILITY_CHECK=${ROOT_READABILITY_CHECK:-0}
ROOT_CHECK_TIMEOUT_SECONDS=${ROOT_CHECK_TIMEOUT_SECONDS:-30}

# ---------------- Editable physics/data parameters -----------
PARTICLE=${PARTICLE:-pi+}
ENERGIES=${ENERGIES:-"1 5 10 20 30 40 50 60 70 80 90 100 110 120"}
SPAD_SIZES=${SPAD_SIZES:-"1x1 5x5 10x10 20x20 50x50 100x100"}
CHANNEL_SIZE=${CHANNEL_SIZE:-1000x1000}
TIME_SLICES=${TIME_SLICES:-"0-8,8-9,9-9.1,9.1-9.2,9.2-9.3,9.3-9.4,9.4-9.5,9.5-9.6,9.6-9.7,9.7-9.8,9.8-9.9,9.9-10,10-10.2,10.2-10.4,10.4-10.6,10.6-10.8,10.8-11,11-12,12-13,13-14,14-15,15-16,16-17,17-18,18-19,19-20,20-21,21-22,22-23,23-24,24-25,25-40"}
TIME_SLICE_COUNT=${TIME_SLICE_COUNT:-$(printf '%s' "$TIME_SLICES" | tr ',' '\n' | awk 'NF {n++} END {print n+0}')}
TIME_SLICE_TAG=${TIME_SLICE_TAG:-ts${TIME_SLICE_COUNT}_$(printf '%s' "$TIME_SLICES" | cksum | awk '{print $1}')}

TRAIN_ROOT_DIR=${TRAIN_ROOT_DIR:-/lustre/work/$USER/pi_train}
PREDICT_ROOT_DIR=${PREDICT_ROOT_DIR:-/lustre/work/$USER/pi_predict}
SPAD_RESULTS_ROOT=${SPAD_RESULTS_ROOT:-/lustre/work/$USER/SPAD_results}
PHOTON_ANALYSIS_DIR=${PHOTON_ANALYSIS_DIR:-$SPAD_RESULTS_ROOT/photon_full_analysis}
CUMULATIVE_DIR=${CUMULATIVE_DIR:-$SPAD_RESULTS_ROOT/cumulative_npy}
CURRENT_TENSOR_ROOT=${CURRENT_TENSOR_ROOT:-$SPAD_RESULTS_ROOT/current_tensors}
NN_ROOT=${NN_ROOT:-$REPO_DIR/NN_Analysis}
MODEL_TEMPLATE=${MODEL_TEMPLATE:-$NN_ROOT/model_copy}
SIM_DIR=${SIM_DIR:-$REPO_DIR/../DREAMSim/sim/build}
SIM_MACRO=${SIM_MACRO:-$SIM_DIR/paramBatch03_single.mac}
SINGULARITY_IMAGE=${SINGULARITY_IMAGE:-/lustre/research/hep/yofeng/SimulationEnv/alma9forgeant4_sbox}

if [ "$TRAIN_ROOT_DIR" = "$PREDICT_ROOT_DIR" ]; then
    echo "ERROR: TRAIN_ROOT_DIR and PREDICT_ROOT_DIR must be different so training and prediction use separate event sets." >&2
    exit 1
fi

# Separate event batches for training and prediction.
# A plain "sbatch run_streamlined_workflow.sh" runs a small end-to-end test by default.
# Set QUICK_TEST=0 for the 14,000-event production setup.
QUICK_TEST=${QUICK_TEST:-1}
if [ "$QUICK_TEST" = "1" ]; then
    TRAIN_SIM_GROUP_SIZE=${TRAIN_SIM_GROUP_SIZE:-28}
    TRAIN_SIM_NJOBS=${TRAIN_SIM_NJOBS:-14}
    PREDICT_SIM_GROUP_SIZE=${PREDICT_SIM_GROUP_SIZE:-28}
    PREDICT_SIM_NJOBS=${PREDICT_SIM_NJOBS:-14}
    NN_EPOCHS=${NN_EPOCHS:-2}
    NN_TRAIN_BATCH_SIZE=${NN_TRAIN_BATCH_SIZE:-8}
    NN_PREDICT_BATCH_SIZE=${NN_PREDICT_BATCH_SIZE:-32}
    NN_WORKERS=${NN_WORKERS:-2}
    NN_VAL_SPLIT=${NN_VAL_SPLIT:-0.50}
    MIN_ROOT_SIZE_KB=${MIN_ROOT_SIZE_KB:-100}
    SIMSPADS_TIMEOUT_SECONDS=${SIMSPADS_TIMEOUT_SECONDS:-120}
    SPAD_JOB_TIME_LIMIT=${SPAD_JOB_TIME_LIMIT:-00:05:00}
else
    TRAIN_SIM_GROUP_SIZE=${TRAIN_SIM_GROUP_SIZE:-14000}
    TRAIN_SIM_NJOBS=${TRAIN_SIM_NJOBS:-140}
    PREDICT_SIM_GROUP_SIZE=${PREDICT_SIM_GROUP_SIZE:-14000}
    PREDICT_SIM_NJOBS=${PREDICT_SIM_NJOBS:-140}
    NN_EPOCHS=${NN_EPOCHS:-20}
    NN_TRAIN_BATCH_SIZE=${NN_TRAIN_BATCH_SIZE:-32}
    NN_PREDICT_BATCH_SIZE=${NN_PREDICT_BATCH_SIZE:-256}
    NN_WORKERS=${NN_WORKERS:-8}
    NN_VAL_SPLIT=${NN_VAL_SPLIT:-0.30}
    MIN_ROOT_SIZE_KB=${MIN_ROOT_SIZE_KB:-1024}
    SIMSPADS_TIMEOUT_SECONDS=${SIMSPADS_TIMEOUT_SECONDS:-3600}
    SPAD_JOB_TIME_LIMIT=${SPAD_JOB_TIME_LIMIT:-01:15:00}
fi
MIN_ROOT_SIZE_KB=${MIN_ROOT_SIZE_KB:-$((MIN_ROOT_SIZE_MB * 1024))}
SIMSPADS_TIMEOUT_SECONDS=${SIMSPADS_TIMEOUT_SECONDS:-600}
SPAD_JOB_TIME_LIMIT=${SPAD_JOB_TIME_LIMIT:-00:20:00}
SIM_PARTITION=${SIM_PARTITION:-nocona}
SIM_MEMORY=${SIM_MEMORY:-16G}
PYTHON_ENV_BIN=${PYTHON_ENV_BIN:-$HOME/miniconda3/envs/dsipm-spad/bin}
if [ ! -x "$PYTHON_ENV_BIN/python3" ] && [ -x "$HOME/miniconda3/envs/base/bin/python3" ]; then
    PYTHON_ENV_BIN="$HOME/miniconda3/envs/base/bin"
fi
SIM_STARTUP_JITTER_SECONDS=${SIM_STARTUP_JITTER_SECONDS:-120}
SIM_ATTEMPTS=${SIM_ATTEMPTS:-3}
SIM_RETRY_SLEEP_SECONDS=${SIM_RETRY_SLEEP_SECONDS:-120}

# SPAD conversion job bookkeeping checks. These do not change the ROOT files read.
SPAD_GROUP_SIZE=${SPAD_GROUP_SIZE:-$TRAIN_SIM_GROUP_SIZE}
SPAD_NJOBS=${SPAD_NJOBS:-$TRAIN_SIM_NJOBS}
SPAD_PARTITION=${SPAD_PARTITION:-nocona}
SPAD_MEMORY_PER_CPU=${SPAD_MEMORY_PER_CPU:-16G}

# Neural network settings.
NN_PARTITION=${NN_PARTITION:-matador}
NN_MEMORY=${NN_MEMORY:-32G}
NN_CPUS=${NN_CPUS:-8}
NN_EPOCHS=${NN_EPOCHS:-50}
NN_TRAIN_BATCH_SIZE=${NN_TRAIN_BATCH_SIZE:-32}
NN_PREDICT_BATCH_SIZE=${NN_PREDICT_BATCH_SIZE:-256}
NN_WORKERS=${NN_WORKERS:-8}
NN_LEARNING_RATE=${NN_LEARNING_RATE:-3e-4}
NN_VAL_SPLIT=${NN_VAL_SPLIT:-0.30}
NN_EPOCH_SAMPLES=${NN_EPOCH_SAMPLES:-0}
NN_MODEL_BASE=${NN_MODEL_BASE:-NN_model_${TIME_SLICE_TAG}_${TRAIN_SIM_GROUP_SIZE}train}
SYNC_NN_TEMPLATE=${SYNC_NN_TEMPLATE:-1}

# Plotting settings.
HIST_ENERGIES=${HIST_ENERGIES:-10,20,80}
MAX_HITMAP_FILES=${MAX_HITMAP_FILES:-0}
EVENT_HITMAP_INDEX=${EVENT_HITMAP_INDEX:-0}

LOG_DIR=${LOG_DIR:-$REPO_DIR/workflow_logs}
ROOT_CHECK_CACHE=${ROOT_CHECK_CACHE:-$LOG_DIR/root_check_cache.tsv}
mkdir -p "$LOG_DIR" "$CUMULATIVE_DIR" "$CURRENT_TENSOR_ROOT"
touch "$ROOT_CHECK_CACHE"

msg() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }

validate_job_layout() {
    local label=$1
    local group_size=$2
    local n_jobs=$3
    local n_energies
    read -ra _preflight_energies <<< "$ENERGIES"
    n_energies=${#_preflight_energies[@]}
    if (( n_energies == 0 )); then
        echo "ERROR: $label has no energies configured." >&2
        exit 1
    fi
    if (( group_size <= 0 || n_jobs <= 0 )); then
        echo "ERROR: $label group size and nJobs must be positive." >&2
        exit 1
    fi
    if (( group_size % n_jobs != 0 )); then
        echo "ERROR: $label group size ($group_size) is not divisible by nJobs ($n_jobs)." >&2
        exit 1
    fi
    if (( n_jobs % n_energies != 0 )); then
        echo "ERROR: $label nJobs ($n_jobs) is not divisible by number of energies ($n_energies)." >&2
        exit 1
    fi
}

preflight_checks() {
    validate_job_layout train "$TRAIN_SIM_GROUP_SIZE" "$TRAIN_SIM_NJOBS"
    validate_job_layout predict "$PREDICT_SIM_GROUP_SIZE" "$PREDICT_SIM_NJOBS"

    if (( SIM_ATTEMPTS < 1 )); then
        echo "ERROR: SIM_ATTEMPTS must be at least 1." >&2
        exit 1
    fi
    if (( SIM_STARTUP_JITTER_SECONDS < 0 || SIM_RETRY_SLEEP_SECONDS < 0 )); then
        echo "ERROR: SIM_STARTUP_JITTER_SECONDS and SIM_RETRY_SLEEP_SECONDS must be nonnegative." >&2
        exit 1
    fi

    if [ "$RUN_GEANT4" = "1" ]; then
        command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is not available." >&2; exit 1; }
        [ -x ./batch_Sims.sh ] || { echo "ERROR: batch_Sims.sh is missing or not executable." >&2; exit 1; }
        [ -x "$SIM_DIR/exampleB4b" ] || { echo "ERROR: Missing executable $SIM_DIR/exampleB4b." >&2; exit 1; }
        [ -s "$SIM_MACRO" ] || { echo "ERROR: Missing simulation macro $SIM_MACRO." >&2; exit 1; }
        [ -e "$SINGULARITY_IMAGE" ] || { echo "ERROR: Missing Singularity image $SINGULARITY_IMAGE." >&2; exit 1; }
    fi
    if [ "$RUN_SPAD_TENSORS" = "1" ]; then
        [ -x ./batch_simSPADs.sh ] || { echo "ERROR: batch_simSPADs.sh is missing or not executable." >&2; exit 1; }
    fi
    if [ "$WAIT_FOR_JOBS" = "1" ]; then
        command -v squeue >/dev/null 2>&1 || { echo "ERROR: squeue is not available but WAIT_FOR_JOBS=1." >&2; exit 1; }
    fi

    msg "Preflight passed: train=${TRAIN_SIM_GROUP_SIZE}/${TRAIN_SIM_NJOBS}, predict=${PREDICT_SIM_GROUP_SIZE}/${PREDICT_SIM_NJOBS}, sim attempts=$SIM_ATTEMPTS, sim memory=$SIM_MEMORY."
}

run_and_collect_job_ids() {
    local label=$1
    shift
    local log_file="$LOG_DIR/${label}_$(date '+%Y%m%d_%H%M%S').log"
    msg "Starting $label"
    "$@" 2>&1 | tee "$log_file" >&2
    awk '/Submitted batch job/ {print $4}' "$log_file" | paste -sd, -
}

wait_for_jobs() {
    local label=$1
    local ids=${2:-}
    if [ "$WAIT_FOR_JOBS" != "1" ] || [ -z "$ids" ]; then
        [ -z "$ids" ] && msg "No SLURM job IDs captured for $label; continuing without a job wait."
        return 0
    fi
    if ! command -v squeue >/dev/null 2>&1; then
        msg "squeue is unavailable; cannot wait for $label jobs."
        return 0
    fi
    msg "Waiting for $label jobs: $ids"
    while any_jobs_in_queue "$ids"; do
        sleep "$SLURM_POLL_SECONDS"
    done
    msg "$label jobs have left the queue."
}

any_jobs_in_queue() {
    local ids=${1:-}
    local id chunk joined
    local count=0
    local -a all_ids=()
    local -a chunk_ids=()

    IFS=',' read -ra all_ids <<< "$ids"
    for id in "${all_ids[@]}"; do
        [ -z "$id" ] && continue
        chunk_ids+=("$id")
        count=$((count + 1))
        if (( count >= SLURM_ID_CHUNK_SIZE )); then
            joined=$(IFS=,; printf '%s' "${chunk_ids[*]}")
            if squeue -h -j "$joined" 2>/dev/null | grep -q .; then
                return 0
            fi
            chunk_ids=()
            count=0
        fi
    done

    if (( ${#chunk_ids[@]} > 0 )); then
        joined=$(IFS=,; printf '%s' "${chunk_ids[*]}")
        if squeue -h -j "$joined" 2>/dev/null | grep -q .; then
            return 0
        fi
    fi
    return 1
}

failed_job_scripts() {
    local ids=$1
    if [ "$WAIT_FOR_JOBS" != "1" ] || [ -z "$ids" ]; then
        return 0
    fi
    if ! command -v sacct >/dev/null 2>&1; then
        msg "sacct is unavailable; skipping automatic failed-job retry check."
        return 0
    fi

    sleep "$SLURM_SETTLE_SECONDS"
    sacct_failed_jobs "$ids" | while IFS="|" read -r job state; do
        state=${state%%+*}
        state=${state%% *}
        case "$state" in
            FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
                case "$job" in
                    Simulations_*)
                        printf "%s\n" "$REPO_DIR/batch_jobs/${job}.sh"
                        ;;
                    *_simSPAD_*)
                        printf "%s\n" "$REPO_DIR/batch_jobs/${job}.sh"
                        ;;
                    NN_train_*)
                        local spad=${job#NN_train_}
                        printf "%s\n" "$NN_ROOT/${spad}_model/NN_Training_${spad}.sh"
                        ;;
                    NN_pred_*)
                        local spad=${job#NN_pred_}
                        printf "%s\n" "$NN_ROOT/${spad}_model/NN_Predict_${spad}.sh"
                        ;;
                    *)
                        msg "No retry script mapping for failed job $job ($state)."
                        ;;
                esac
                ;;
        esac
    done | sort -u
}

sacct_failed_jobs() {
    local ids=${1:-}
    local id joined
    local count=0
    local -a all_ids=()
    local -a chunk_ids=()

    IFS=',' read -ra all_ids <<< "$ids"
    for id in "${all_ids[@]}"; do
        [ -z "$id" ] && continue
        chunk_ids+=("$id")
        count=$((count + 1))
        if (( count >= SLURM_ID_CHUNK_SIZE )); then
            joined=$(IFS=,; printf '%s' "${chunk_ids[*]}")
            sacct -X -n -P -j "$joined" --format=JobName%200,State%40 2>/dev/null || true
            chunk_ids=()
            count=0
        fi
    done

    if (( ${#chunk_ids[@]} > 0 )); then
        joined=$(IFS=,; printf '%s' "${chunk_ids[*]}")
        sacct -X -n -P -j "$joined" --format=JobName%200,State%40 2>/dev/null || true
    fi
}

resubmit_failed_scripts() {
    local scripts=$1
    local new_ids=""
    local script id
    while IFS= read -r script; do
        [ -z "$script" ] && continue
        if [ ! -f "$script" ]; then
            msg "Cannot retry missing script: $script"
            continue
        fi
        msg "Retrying failed job script: $script"
        id=$(cd "$(dirname "$script")" && sbatch "$(basename "$script")" | awk '/Submitted batch job/ {print $4}')
        new_ids=${new_ids:+$new_ids,}$id
    done <<< "$scripts"
    printf "%s\n" "$new_ids"
}

wait_for_jobs_with_retries() {
    local label=$1
    local ids=${2:-}
    local attempt=0
    local failed_scripts=""

    while true; do
        wait_for_jobs "$label" "$ids"
        failed_scripts=$(failed_job_scripts "$ids")
        if [ -z "$failed_scripts" ]; then
            return 0
        fi

        attempt=$((attempt + 1))
        if (( attempt > MAX_JOB_RETRIES )); then
            echo "ERROR: $label still has failed jobs after $MAX_JOB_RETRIES retries:" >&2
            printf "%s\n" "$failed_scripts" >&2
            exit 1
        fi

        msg "$label has failed jobs; retry attempt $attempt of $MAX_JOB_RETRIES."
        ids=$(resubmit_failed_scripts "$failed_scripts")
        if [ -z "$ids" ]; then
            echo "ERROR: Could not resubmit failed $label jobs." >&2
            exit 1
        fi
    done
}

require_npz() {
    local dir=$1
    if ! find "$dir" \( -type f -o -type l \) -name '*.npz' -print -quit | grep -q .; then
        echo "ERROR: No .npz tensors found under $dir" >&2
        exit 1
    fi
}

root_energy_args() {
    local root_dir=$1
    find "$root_dir" -maxdepth 1 -type f -name '*.root' | sort | while read -r f; do
        local base energy
        base=$(basename "$f")
        energy=$(printf '%s\n' "$base" | sed -n 's/.*_\([0-9][0-9.]*\)GeV_.*/\1/p')
        if [ -n "$energy" ]; then
            printf '%s:%s\n' "$f" "$energy"
        fi
    done
}


root_file_retry_script() {
    local root_file=$1
    local base energy job_index
    base=$(basename "$root_file")
    energy=$(printf '%s\n' "$base" | sed -n 's/.*_\([0-9][0-9.]*\)GeV_\([0-9][0-9]*\)_.*/\1/p')
    job_index=$(printf '%s\n' "$base" | sed -n 's/.*_\([0-9][0-9.]*\)GeV_\([0-9][0-9]*\)_.*/\2/p')
    if [ -z "$energy" ] || [ -z "$job_index" ]; then
        return 1
    fi
    printf '%s\n' "$REPO_DIR/batch_jobs/Simulations_${job_index}_${energy}.sh"
}

root_file_cache_key() {
    local path=$1
    stat -c 'v2\t%n\t%s\t%Y' "$path" 2>/dev/null
}

root_file_failure_reason() {
    local path=$1
    local size min_bytes key
    min_bytes=$((MIN_ROOT_SIZE_KB * 1024))

    if [ ! -f "$path" ]; then
        printf 'missing file'
        return 1
    fi

    size=$(stat -c '%s' "$path" 2>/dev/null || printf '0')
    if (( size < min_bytes )); then
        printf 'too small: %s bytes < %s bytes' "$size" "$min_bytes"
        return 1
    fi

    if [ "$ROOT_READABILITY_CHECK" != "1" ]; then
        return 0
    fi

    key=$(root_file_cache_key "$path" || true)
    if [ -n "$key" ] && grep -Fxq "$key" "$ROOT_CHECK_CACHE" 2>/dev/null; then
        return 0
    fi

    if timeout "$ROOT_CHECK_TIMEOUT_SECONDS" python3 - "$path" >/dev/null 2>&1 <<'PYROOTONE'
import sys
import ROOT

ROOT.gROOT.SetBatch(True)
path = sys.argv[1]
f = ROOT.TFile(path, "READ")
if not f or f.IsZombie():
    raise SystemExit(1)
tree = f.Get("tree")
if not tree:
    f.Close()
    raise SystemExit(1)
entries = int(tree.GetEntries())
if entries <= 0:
    f.Close()
    raise SystemExit(1)
if tree.GetEntry(0) <= 0:
    f.Close()
    raise SystemExit(1)
f.Close()
PYROOTONE
    then
        [ -n "$key" ] && printf '%s\n' "$key" >> "$ROOT_CHECK_CACHE"
        return 0
    fi

    printf 'ROOT unreadable, zombie, missing tree, or timed out after %ss' "$ROOT_CHECK_TIMEOUT_SECONDS"
    return 1
}

bad_root_files() {
    local root_dir=$1
    local path reason
    while IFS= read -r -d '' path; do
        if reason=$(root_file_failure_reason "$path"); then
            continue
        fi
        printf '%s|%s\n' "$path" "$reason"
    done < <(find "$root_dir" -maxdepth 1 -type f -name '*.root' -print0 | sort -z)
}

validate_root_files_with_retries() {
    local label=$1
    local root_dir=$2
    local attempt=0
    local bad_lines scripts ids bad_path reason script stamp

    while true; do
        bad_lines=$(bad_root_files "$root_dir")
        if [ -z "$bad_lines" ]; then
            msg "$label ROOT files passed readability checks."
            return 0
        fi

        attempt=$((attempt + 1))
        if (( attempt > MAX_JOB_RETRIES )); then
            echo "ERROR: $label still has unreadable ROOT files after $MAX_JOB_RETRIES retries:" >&2
            printf "%s\n" "$bad_lines" >&2
            exit 1
        fi

        msg "$label has unreadable ROOT files; retry attempt $attempt of $MAX_JOB_RETRIES."
        scripts=""
        stamp=$(date '+%Y%m%d_%H%M%S')
        while IFS='|' read -r bad_path reason; do
            [ -z "$bad_path" ] && continue
            msg "Bad ROOT file: $bad_path ($reason)"
            if script=$(root_file_retry_script "$bad_path"); then
                scripts=${scripts:+$scripts$'\n'}$script
                if [ -f "$bad_path" ]; then
                    mv "$bad_path" "${bad_path}.bad_${stamp}"
                fi
            else
                echo "ERROR: Could not map bad ROOT file to a retry script: $bad_path" >&2
                exit 1
            fi
        done <<< "$bad_lines"

        scripts=$(printf "%s\n" "$scripts" | sort -u)
        ids=$(resubmit_failed_scripts "$scripts")
        if [ -z "$ids" ]; then
            echo "ERROR: Could not resubmit bad ROOT file jobs for $label." >&2
            exit 1
        fi
        wait_for_jobs_with_retries "$label root_file_retry" "$ids"
    done
}


missing_sim_jobs() {
    local root_dir=$1
    local group_size=$2
    local n_jobs=$3
    local energies=$4
    local particle=$5
    local n_energies job_size jobs_per_energy energy idx pattern path reason
    local -a matches=()

    read -ra energy_list <<< "$energies"
    n_energies=${#energy_list[@]}
    if (( n_energies == 0 )); then
        echo "ERROR: No energies configured" >&2
        return 1
    fi
    if (( group_size % n_jobs != 0 )); then
        echo "ERROR: Group size not divisible by number of jobs" >&2
        return 1
    fi
    if (( n_jobs % n_energies != 0 )); then
        echo "ERROR: Number of jobs not divisible by number of energies" >&2
        return 1
    fi

    job_size=$((group_size / n_jobs))
    jobs_per_energy=$((n_jobs / n_energies))

    for energy in "${energy_list[@]}"; do
        for (( idx=0; idx<jobs_per_energy; idx++ )); do
            pattern="*${job_size}events_${energy}GeV_${idx}_${particle}*.root"
            matches=()
            while IFS= read -r -d '' path; do
                matches+=("$path")
            done < <(find "$root_dir" -maxdepth 1 -type f -name "$pattern" -print0 | sort -z)

            good=0
            bad_paths=""
            for path in "${matches[@]}"; do
                if reason=$(root_file_failure_reason "$path"); then
                    good=1
                    break
                fi
                bad_paths=${bad_paths:+$bad_paths,}$path
            done

            if (( good == 1 )); then
                continue
            fi
            if (( ${#matches[@]} > 0 )); then
                printf '%s|%s|bad|%s\n' "$energy" "$idx" "$bad_paths"
            else
                printf '%s|%s|missing|-\n' "$energy" "$idx"
            fi
        done
    done
}

run_geant4_stage() {
    local label=$1
    local root_dir=$2
    local group_size=$3
    local n_jobs=$4
    local attempt=0
    local missing_lines only_jobs ids energy idx reason paths bad_path stamp

    mkdir -p "$root_dir"
    while true; do
        missing_lines=$(missing_sim_jobs "$root_dir" "$group_size" "$n_jobs" "$ENERGIES" "$PARTICLE")
        if [ -z "$missing_lines" ]; then
            msg "$label ROOT files already exist, are at least ${MIN_ROOT_SIZE_KB} KB, and passed configured checks; moving ahead."
            return 0
        fi

        attempt=$((attempt + 1))
        if (( attempt > MAX_JOB_RETRIES + 1 )); then
            echo "ERROR: $label still has missing/bad ROOT files after $MAX_JOB_RETRIES retries:" >&2
            printf "%s\n" "$missing_lines" >&2
            exit 1
        fi

        only_jobs=""
        stamp=$(date '+%Y%m%d_%H%M%S')
        while IFS='|' read -r energy idx reason paths; do
            [ -z "$energy" ] && continue
            only_jobs=${only_jobs:+$only_jobs }${energy}:${idx}
            msg "$label needs simulation ${energy}:${idx} ($reason)."
            if [ "$reason" = "bad" ] && [ "$paths" != "-" ]; then
                IFS=',' read -ra bad_paths <<< "$paths"
                for bad_path in "${bad_paths[@]}"; do
                    [ -f "$bad_path" ] && mv "$bad_path" "${bad_path}.bad_${stamp}"
                done
            fi
        done <<< "$missing_lines"

        msg "$label submitting only missing/bad simulations: $only_jobs"
        ids=$(run_and_collect_job_ids "$label" env \
            DEST_DIR="$root_dir" PARTICLE="$PARTICLE" Energies="$ENERGIES" \
            Group_Size="$group_size" nJobs="$n_jobs" ONLY_JOBS="$only_jobs" \
            SIM_DIR="$SIM_DIR" SIM_MACRO="$SIM_MACRO" SINGULARITY_IMAGE="$SINGULARITY_IMAGE" \
            PARTITION="$SIM_PARTITION" MEMORY="$SIM_MEMORY" \
            SIM_STARTUP_JITTER_SECONDS="$SIM_STARTUP_JITTER_SECONDS" \
            SIM_ATTEMPTS="$SIM_ATTEMPTS" SIM_RETRY_SLEEP_SECONDS="$SIM_RETRY_SLEEP_SECONDS" ./batch_Sims.sh)
        if [ -z "$ids" ]; then
            echo "ERROR: $label did not submit any jobs for: $only_jobs" >&2
            exit 1
        fi
        wait_for_jobs_with_retries "$label" "$ids"
    done
}

spad_expected_events() {
    local group_size=$1
    local n_jobs=$2
    if (( n_jobs <= 0 )); then
        printf '0\n'
    else
        printf '%s\n' $((group_size / n_jobs))
    fi
}

spad_output_dir_for_root() {
    local root_file=$1
    local tag=$2
    local spad=$3
    local base energy run
    base=$(basename "$root_file")
    energy=$(printf '%s\n' "$base" | sed -n 's/.*_\([0-9][0-9.]*\)GeV_.*/\1/p')
    run=$(printf '%s\n' "$base" | sed -n 's/.*GeV_\([0-9][0-9]*\)_.*/\1/p')
    if [ -z "$energy" ] || [ -z "$run" ]; then
        return 1
    fi
    printf '%s/%s_%sGeV_run%s_SPAD%s_CH%s\n' "$SPAD_RESULTS_ROOT/${tag}_${spad}" "$tag" "$energy" "$run" "$spad" "$CHANNEL_SIZE"
}

spad_failure_marker_for_root() {
    local root_file=$1
    local tag=$2
    local spad=$3
    local base
    base=$(basename "$root_file")
    printf '%s/.failed_roots/%s_SPAD%s.failed\n' "$SPAD_RESULTS_ROOT/${tag}_${spad}" "$base" "$spad"
}

spad_failure_marker_reason() {
    local root_file=$1
    local tag=$2
    local spad=$3
    local marker marker_key current_key exit_code
    marker=$(spad_failure_marker_for_root "$root_file" "$tag" "$spad")
    [ -s "$marker" ] || return 1
    marker_key=$(sed -n '1p' "$marker" 2>/dev/null || true)
    current_key=$(root_file_cache_key "$root_file" || true)
    [ -n "$marker_key" ] && [ "$marker_key" = "$current_key" ] || return 1
    exit_code=$(sed -n 's/^exit_code=//p' "$marker" 2>/dev/null | tail -1)
    printf 'previous SPAD conversion failed or timed out%s' "${exit_code:+ (exit $exit_code)}"
    return 0
}


spad_output_failure_reason() {
    local out_dir=$1
    local expected_events=$2
    local expected_channels=$3
    local expected_time_slices=$4
    local stats slices_file npz_count line_count sample_npz actual_channels actual_time_slices npz_time_check

    stats="$out_dir/photon_stats.csv"
    if [ ! -s "$stats" ]; then
        printf 'missing photon_stats.csv'
        return 1
    fi
    slices_file="$out_dir/time_slices.txt"
    if [ ! -s "$slices_file" ]; then
        printf 'missing time_slices.txt'
        return 1
    fi
    actual_time_slices=$(tr -d '\r\n' < "$slices_file")
    if [ "$actual_time_slices" != "$expected_time_slices" ]; then
        printf 'time slices changed'
        return 1
    fi
    if [ ! -d "$out_dir/npy" ]; then
        printf 'missing npy directory'
        return 1
    fi
    npz_count=$(find "$out_dir/npy" -maxdepth 1 -type f -name '*.npz' ! -name '*_dup*.npz' 2>/dev/null | wc -l)
    if (( npz_count <= 0 )); then
        printf 'missing npz tensors'
        return 1
    fi
    if (( expected_events > 0 && npz_count != expected_events )); then
        printf 'has %s/%s npz tensors' "$npz_count" "$expected_events"
        return 1
    fi
    sample_npz=$(find "$out_dir/npy" -maxdepth 1 -type f -name '*.npz' ! -name '*_dup*.npz' -print -quit 2>/dev/null || true)
    if [ -n "$sample_npz" ] && [ -n "$expected_channels" ] && (( expected_channels > 0 )); then
        npz_time_check=$(python3 - "$sample_npz" "$expected_time_slices" <<'PYCHANNELS'
import sys
import numpy as np
path = sys.argv[1]
expected_spec = sys.argv[2]

def parse(spec):
    pairs = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, high = part.split("-", 1)
        pairs.append((float(low), float(high)))
    return np.asarray(pairs, dtype=np.float32)

with np.load(path, allow_pickle=False) as z:
    x = z["x"]
    actual_channels = int(x.shape[0])
    if "time_slices" not in z:
        print(f"ERR|tensor missing time_slices metadata ({actual_channels} channels)")
        sys.exit(0)
    actual_slices = np.asarray(z["time_slices"], dtype=np.float32)

expected_slices = parse(expected_spec)
if actual_slices.shape != expected_slices.shape or not np.allclose(actual_slices, expected_slices, rtol=0.0, atol=1.0e-6):
    print(f"ERR|tensor time_slices metadata mismatch ({actual_channels} channels)")
else:
    print(f"OK|{actual_channels}")
PYCHANNELS
)
        if [[ "$npz_time_check" == ERR\|* ]]; then
            printf '%s' "${npz_time_check#ERR|}"
            return 1
        fi
        actual_channels=${npz_time_check#OK|}
        if [ "$actual_channels" != "$expected_channels" ]; then
            printf 'tensor has %s time slices, expected %s' "$actual_channels" "$expected_channels"
            return 1
        fi
    fi
    line_count=$(wc -l < "$stats" 2>/dev/null || printf '0')
    if (( expected_events > 0 && line_count != expected_events + 1 )); then
        printf 'has %s/%s photon stat rows' "$((line_count - 1))" "$expected_events"
        return 1
    fi
    return 0
}

missing_spad_outputs() {
    local root_dir=$1
    local tag=$2
    local spad=$3
    local expected_events=$4
    local group_size=$5
    local n_jobs=$6
    local n_energies job_size jobs_per_energy energy idx pattern root_file path reason out_dir out_reason marker_reason
    local -a energy_list matches

    read -ra energy_list <<< "$ENERGIES"
    n_energies=${#energy_list[@]}
    if (( n_energies == 0 || group_size % n_jobs != 0 || n_jobs % n_energies != 0 )); then
        printf '%s|ROOT bad workflow simulation settings|-\n' "$root_dir"
        return 0
    fi
    job_size=$((group_size / n_jobs))
    jobs_per_energy=$((n_jobs / n_energies))

    for energy in "${energy_list[@]}"; do
        for (( idx=0; idx<jobs_per_energy; idx++ )); do
            pattern="*${job_size}events_${energy}GeV_${idx}_${PARTICLE}*.root"
            matches=()
            while IFS= read -r -d '' path; do
                matches+=("$path")
            done < <(find "$root_dir" -maxdepth 1 -type f -name "$pattern" -print0 | sort -z)

            root_file=""
            for path in "${matches[@]}"; do
                if reason=$(root_file_failure_reason "$path"); then
                    root_file="$path"
                    break
                fi
            done

            if [ -z "$root_file" ]; then
                if (( ${#matches[@]} > 0 )); then
                    printf '%s|ROOT no good current ROOT for %s:%s (%s bad match(es))|-\n' "$root_dir" "$energy" "$idx" "${#matches[@]}"
                else
                    printf '%s|ROOT missing current ROOT for %s:%s|-\n' "$root_dir" "$energy" "$idx"
                fi
                continue
            fi

            if marker_reason=$(spad_failure_marker_reason "$root_file" "$tag" "$spad"); then
                printf '%s|ROOT %s|-\n' "$root_file" "$marker_reason"
                continue
            fi

            if out_dir=$(spad_output_dir_for_root "$root_file" "$tag" "$spad"); then
                if out_reason=$(spad_output_failure_reason "$out_dir" "$expected_events" "$TIME_SLICE_COUNT" "$TIME_SLICES"); then
                    continue
                fi
                printf '%s|%s|%s\n' "$root_file" "$out_reason" "$out_dir"
            else
                printf '%s|could not parse energy/run|-\n' "$root_file"
            fi
        done
    done
}

run_spad_stage() {
    local label=$1
    local tag=$2
    local root_dir=$3
    local spad=$4
    local expected_events=$5
    local sim_group_size=$6
    local sim_n_jobs=$7
    local attempt=0
    local missing_lines ids only_roots root_file reason out_dir stamp root_bad

    while true; do
        missing_lines=$(missing_spad_outputs "$root_dir" "$tag" "$spad" "$expected_events" "$sim_group_size" "$sim_n_jobs")
        if [ -z "$missing_lines" ]; then
            msg "$label outputs already exist and passed checks; moving ahead."
            return 0
        fi

        attempt=$((attempt + 1))
        if (( attempt > MAX_JOB_RETRIES + 1 )); then
            echo "ERROR: $label still has missing/bad SPAD outputs after $MAX_JOB_RETRIES retries:" >&2
            printf '%s\n' "$missing_lines" >&2
            exit 1
        fi

        only_roots=""
        root_bad=0
        stamp=$(date '+%Y%m%d_%H%M%S')
        while IFS='|' read -r root_file reason out_dir; do
            [ -z "$root_file" ] && continue
            msg "$label needs $(basename "$root_file") ($reason)."
            case "$reason" in
                ROOT*)
                    root_bad=1
                    if [ -f "$root_file" ]; then
                        msg "$label marking ROOT for regeneration: $root_file"
                        mv "$root_file" "${root_file}.bad_${stamp}"
                    fi
                    continue
                    ;;
            esac
            if [ "$out_dir" != "-" ] && [ -d "$out_dir" ]; then
                mv "$out_dir" "${out_dir}.bad_${stamp}"
            fi
            only_roots=${only_roots:+$only_roots }$root_file
        done <<< "$missing_lines"

        if (( root_bad == 1 )); then
            msg "$label found bad ROOT inputs; regenerating needed ROOT files before retrying SPAD conversion."
            run_geant4_stage "geant4_${tag}_from_${label}" "$root_dir" "$sim_group_size" "$sim_n_jobs"
            continue
        fi

        ids=$(run_and_collect_job_ids "$label" env \
            INPUT_DIR="$root_dir" DEST_DIR="$SPAD_RESULTS_ROOT/${tag}_${spad}" \
            OUTPUT_TAG="$tag" SPAD_Size="$spad" Channel_Size="$CHANNEL_SIZE" \
            Energies="$ENERGIES" Group_Size="$sim_group_size" nJobs="$sim_n_jobs" \
            ONLY_ROOTS="$only_roots" TIME_SLICES="$TIME_SLICES" \
            PARTITION="$SPAD_PARTITION" MEMORY_PER_CPU="$SPAD_MEMORY_PER_CPU" \
            SIMSPADS_TIMEOUT_SECONDS="$SIMSPADS_TIMEOUT_SECONDS" SPAD_JOB_TIME_LIMIT="$SPAD_JOB_TIME_LIMIT" \
            PYTHON_ENV_BIN="$PYTHON_ENV_BIN" ./batch_simSPADs.sh)
        if [ -z "$ids" ]; then
            echo "ERROR: $label did not submit any SPAD jobs." >&2
            exit 1
        fi
        wait_for_jobs_with_retries "$label" "$ids"
        sleep "$SLURM_SETTLE_SECONDS"
    done
}

wait_for_photon_stats_ready() {
    local tag=$1
    local expected_total=$2
    local count
    count=$(find "$SPAD_RESULTS_ROOT" -path "*.bad_*" -prune -o -path "*/${tag}_*/*/photon_stats.csv" -print 2>/dev/null | wc -l)
    if (( expected_total > 0 && count < expected_total )); then
        echo "ERROR: Only found $count/$expected_total ${tag} photon_stats.csv files after SPAD checks." >&2
        exit 1
    fi
    msg "Found $count ${tag} photon_stats.csv files."
}

prepare_current_tensor_view() {
    local tensor_dir=$1
    local spad=$2
    local tag=$3
    local group_size=$4
    local n_jobs=$5
    local view_dir="$CURRENT_TENSOR_ROOT/${tag}_${spad}_${TIME_SLICE_TAG}"
    require_npz "$tensor_dir"
    python3 - "$tensor_dir" "$view_dir" "$TIME_SLICES" "$ENERGIES" "$group_size" "$n_jobs" "$spad" "$CHANNEL_SIZE" "$tag" <<'PY'
import glob, os, shutil, sys
import numpy as np
src, view, spec, energies_s, group_size_s, n_jobs_s, spad, channel, tag = sys.argv[1:10]
group_size = int(group_size_s)
n_jobs = int(n_jobs_s)
energies = energies_s.split()
if not energies:
    raise SystemExit('No energies supplied')
if group_size % n_jobs != 0 or n_jobs % len(energies) != 0:
    raise SystemExit('Bad group/n_jobs/energies layout')
job_size = group_size // n_jobs
jobs_per_energy = n_jobs // len(energies)

def parse(spec):
    pairs = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        low, high = part.split('-', 1)
        pairs.append((float(low), float(high)))
    return np.asarray(pairs, dtype=np.float32)

expected = parse(spec)
if os.path.exists(view):
    shutil.rmtree(view)
os.makedirs(view, exist_ok=True)
kept = 0
skipped = 0
missing_dirs = 0
for energy in energies:
    for idx in range(jobs_per_energy):
        out_dir = os.path.join(src, f'{tag}_{energy}GeV_run{idx}_SPAD{spad}_CH{channel}')
        npy_dir = os.path.join(out_dir, 'npy')
        if not os.path.isdir(npy_dir):
            missing_dirs += 1
            continue
        for path in sorted(glob.glob(os.path.join(npy_dir, '*.npz'))):
            if '.bad_' in path or '_dup' in os.path.basename(path):
                continue
            try:
                with np.load(path, allow_pickle=False) as z:
                    x_shape = z['x'].shape
                    if x_shape[0] != expected.shape[0] or 'time_slices' not in z:
                        skipped += 1
                        continue
                    actual = np.asarray(z['time_slices'], dtype=np.float32)
                    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
                        skipped += 1
                        continue
            except Exception:
                skipped += 1
                continue
            rel = os.path.relpath(path, src)
            dest = os.path.join(view, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.symlink(os.path.abspath(path), dest)
            kept += 1
if kept == 0:
    raise SystemExit(f'No expected current tensors in {src}; missing_dirs {missing_dirs}, skipped {skipped}')
print(f'current tensor view {view}: kept {kept}, skipped {skipped}, missing_dirs {missing_dirs}', file=sys.stderr)
PY
    printf '%s\n' "$view_dir"
}


make_cumulative_npy() {
    local tensor_dir=$1
    local spad=$2
    local tag=$3
    require_npz "$tensor_dir"
    python3 - "$tensor_dir" "$CUMULATIVE_DIR/cumulative_${tag}_${spad}.npy" "$TIME_SLICES" <<'PY'
import glob, math, os, sys
import numpy as np
root, out, spec = sys.argv[1], sys.argv[2], sys.argv[3]

def parse(spec):
    pairs = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        low, high = part.split('-', 1)
        pairs.append((float(low), float(high)))
    return np.asarray(pairs, dtype=np.float32)

expected = parse(spec)
acc = None
n = 0
skipped = 0
for path in sorted(glob.glob(os.path.join(root, '**', '*.npz'), recursive=True)):
    if '.bad_' in path or '_dup' in os.path.basename(path):
        continue
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z['x'], dtype=np.float64)
        if x.shape[0] != expected.shape[0] or 'time_slices' not in z:
            skipped += 1
            continue
        actual = np.asarray(z['time_slices'], dtype=np.float32)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
            skipped += 1
            continue
        lnN = float(z['lnN'])
    counts = x * math.exp(lnN)
    if acc is None:
        acc = np.zeros_like(counts, dtype=np.float64)
    acc += counts
    n += 1
if acc is None:
    raise SystemExit(f'No current TIME_SLICES tensors found under {root}; skipped {skipped} stale tensors')
os.makedirs(os.path.dirname(out), exist_ok=True)
np.save(out, acc.astype(np.float32))
print(f'wrote {out} from {n} events with shape {acc.shape}; skipped {skipped} stale tensors')
PY
}


preflight_checks

if [ "$RUN_GEANT4" = "1" ]; then
    run_geant4_stage geant4_train "$TRAIN_ROOT_DIR" "$TRAIN_SIM_GROUP_SIZE" "$TRAIN_SIM_NJOBS"
    run_geant4_stage geant4_predict "$PREDICT_ROOT_DIR" "$PREDICT_SIM_GROUP_SIZE" "$PREDICT_SIM_NJOBS"
fi

if [ "$RUN_SPAD_TENSORS" = "1" ]; then
    train_expected_events=$(spad_expected_events "$TRAIN_SIM_GROUP_SIZE" "$TRAIN_SIM_NJOBS")
    predict_expected_events=$(spad_expected_events "$PREDICT_SIM_GROUP_SIZE" "$PREDICT_SIM_NJOBS")
    for spad in $SPAD_SIZES; do
        run_spad_stage "spad_train_${spad}" train "$TRAIN_ROOT_DIR" "$spad" "$train_expected_events" "$TRAIN_SIM_GROUP_SIZE" "$TRAIN_SIM_NJOBS"
    done
    for spad in $SPAD_SIZES; do
        run_spad_stage "spad_predict_${spad}" predict "$PREDICT_ROOT_DIR" "$spad" "$predict_expected_events" "$PREDICT_SIM_GROUP_SIZE" "$PREDICT_SIM_NJOBS"
    done
fi

if [ "$RUN_PHOTON_FULL_ANALYSIS" = "1" ]; then
    read -ra _spad_list <<< "$SPAD_SIZES"
    expected_predict_stats=$((PREDICT_SIM_NJOBS * ${#_spad_list[@]}))
    wait_for_photon_stats_ready predict "$expected_predict_stats"
    photon_stat_roots=()
    for spad in $SPAD_SIZES; do
        photon_stat_roots+=("$SPAD_RESULTS_ROOT/predict_${spad}")
    done
    msg "Aggregating photon statistics from parallel SPAD tensor jobs into $PHOTON_ANALYSIS_DIR"
    python3 aggregate_photon_stats.py "$PHOTON_ANALYSIS_DIR" "${photon_stat_roots[@]}"
fi

if [ "$RUN_NN_TRAIN" = "1" ]; then
    train_ids_all=""
    for spad in $SPAD_SIZES; do
        train_dir=$(prepare_current_tensor_view "$SPAD_RESULTS_ROOT/train_${spad}" "$spad" train "$TRAIN_SIM_GROUP_SIZE" "$TRAIN_SIM_NJOBS")
        model_dir="$NN_ROOT/${spad}_model"
        if [ ! -d "$model_dir" ]; then
            cp -a "$MODEL_TEMPLATE" "$model_dir"
        elif [ "$SYNC_NN_TEMPLATE" = "1" ]; then
            cp "$MODEL_TEMPLATE"/{train.py,dataset.py,model.py,trainNN.sh,predictNN.sh,plot_nn_outputs.py} "$model_dir"/
        fi
        ids=$(cd "$model_dir" && run_and_collect_job_ids "nn_train_${spad}" env \
            SPAD_SIZE="$spad" TENSOR_DIR="$train_dir" MODEL_BASE="$NN_MODEL_BASE" TIME_SLICES="$TIME_SLICES" \
            EPOCHS="$NN_EPOCHS" BATCH_SIZE="$NN_TRAIN_BATCH_SIZE" WORKERS="$NN_WORKERS" \
            LEARNING_RATE="$NN_LEARNING_RATE" VAL_SPLIT="$NN_VAL_SPLIT" \
            EPOCH_SAMPLES="$NN_EPOCH_SAMPLES" \
            PARTITION="$NN_PARTITION" MEMORY="$NN_MEMORY" CPUS="$NN_CPUS" \
            PYTHON_ENV_BIN="$PYTHON_ENV_BIN" bash ./trainNN.sh)
        train_ids_all=${train_ids_all:+$train_ids_all,}$ids
    done
    wait_for_jobs_with_retries nn_train "$train_ids_all"
fi

if [ "$RUN_NN_PREDICT" = "1" ]; then
    predict_ids_all=""
    for spad in $SPAD_SIZES; do
        pred_dir=$(prepare_current_tensor_view "$SPAD_RESULTS_ROOT/predict_${spad}" "$spad" predict "$PREDICT_SIM_GROUP_SIZE" "$PREDICT_SIM_NJOBS")
        model_dir="$NN_ROOT/${spad}_model"
        checkpoint="$model_dir/${NN_MODEL_BASE}_${spad}/best_latest.ckpt"
        if [ ! -f "$checkpoint" ]; then
            echo "ERROR: Missing checkpoint for $spad: $checkpoint" >&2
            exit 1
        fi
        ids=$(cd "$model_dir" && run_and_collect_job_ids "nn_predict_${spad}" env \
            SPAD_SIZE="$spad" TENSOR_DIR="$pred_dir" MODEL_BASE="$NN_MODEL_BASE" TIME_SLICES="$TIME_SLICES" \
            CHECKPOINT="$checkpoint" PRED_CSV="predictions_${spad}.csv" \
            BATCH_SIZE="$NN_PREDICT_BATCH_SIZE" WORKERS="$NN_WORKERS" \
            PARTITION="$NN_PARTITION" MEMORY="$NN_MEMORY" CPUS="$NN_CPUS" \
            PYTHON_ENV_BIN="$PYTHON_ENV_BIN" bash ./predictNN.sh)
        predict_ids_all=${predict_ids_all:+$predict_ids_all,}$ids
    done
    wait_for_jobs_with_retries nn_predict "$predict_ids_all"
fi

if [ "$RUN_CUMULATIVE_NPY" = "1" ]; then
    for spad in $SPAD_SIZES; do
        train_view=$(prepare_current_tensor_view "$SPAD_RESULTS_ROOT/train_${spad}" "$spad" train "$TRAIN_SIM_GROUP_SIZE" "$TRAIN_SIM_NJOBS")
        predict_view=$(prepare_current_tensor_view "$SPAD_RESULTS_ROOT/predict_${spad}" "$spad" predict "$PREDICT_SIM_GROUP_SIZE" "$PREDICT_SIM_NJOBS")
        make_cumulative_npy "$train_view" "$spad" train
        make_cumulative_npy "$predict_view" "$spad" predict
    done
fi

if [ "$RUN_PLOTS" = "1" ]; then
    for spad in $SPAD_SIZES; do
        msg "Plotting photon/NN outputs and cumulative hit maps for $spad"
        python3 plot.py "$PHOTON_ANALYSIS_DIR" \
            --nn-root "$NN_ROOT" \
            --tensor-root "$(prepare_current_tensor_view "$SPAD_RESULTS_ROOT/predict_${spad}" "$spad" predict "$PREDICT_SIM_GROUP_SIZE" "$PREDICT_SIM_NJOBS")" \
            --spads "$(printf '%s' "$SPAD_SIZES" | sed 's/x[0-9]*//g; s/ /,/g')" \
            --hist-energies "$HIST_ENERGIES" \
            --time-slices "$TIME_SLICES" \
            --event-index "$EVENT_HITMAP_INDEX" \
            --max-hitmap-files "$MAX_HITMAP_FILES"
    done
fi

msg "Streamlined workflow complete."
msg "Training tensors:   $SPAD_RESULTS_ROOT/train_<SPAD_SIZE>"
msg "Prediction tensors: $SPAD_RESULTS_ROOT/predict_<SPAD_SIZE>"
msg "Cumulative npy:     $CUMULATIVE_DIR"
msg "Photon analysis:    $PHOTON_ANALYSIS_DIR"
msg "Plots:              $PHOTON_ANALYSIS_DIR/plots_mpl"
msg "NN models/results:  $NN_ROOT/<SPAD_SIZE>_model/${NN_MODEL_BASE}_<SPAD_SIZE>"
