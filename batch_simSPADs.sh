#!/bin/bash

home_dir=$PWD
sim_dir=${home_dir}/../DREAMSim/sim/build
input_dir=${INPUT_DIR:-/lustre/work/$USER/pi_train}      # Matches batch_Sims.sh by default
OUTPUT_TAG=${OUTPUT_TAG:-train}
SPAD_Size=${SPAD_Size:-200x200}
Channel_Size=${Channel_Size:-1000x1000}
TIME_SLICES=${TIME_SLICES:-"0-8,8-9,9-9.1,9.1-9.2,9.2-9.3,9.3-9.4,9.4-9.5,9.5-9.6,9.6-9.7,9.7-9.8,9.8-9.9,9.9-10,10-10.2,10.2-10.4,10.4-10.6,10.6-10.8,10.8-11,11-12,12-13,13-14,14-15,15-16,16-17,17-18,18-19,19-20,20-21,21-22,22-23,23-24,24-25,25-40"}
dest_dir=${DEST_DIR:-/lustre/work/$USER/SPAD_results/${OUTPUT_TAG}_${SPAD_Size}}

mkdir -p "${dest_dir}"
mkdir -p batch_jobs
mkdir -p batch_jobs/LOGDIR
cd batch_jobs

# --------- Adjustable parameters -------------
particle="pi+"

# SPAD size for deadtime effects
# Override SPAD_Size and Channel_Size at submission time if needed.
PARTITION=${PARTITION:-nocona}
MEMORY_PER_CPU=${MEMORY_PER_CPU:-16G}
SIMSPADS_TIMEOUT_SECONDS=${SIMSPADS_TIMEOUT_SECONDS:-600}
SPAD_JOB_TIME_LIMIT=${SPAD_JOB_TIME_LIMIT:-00:20:00}
PYTHON_ENV_BIN=${PYTHON_ENV_BIN:-$HOME/miniconda3/envs/dsipm-spad/bin}
if [ ! -x "$PYTHON_ENV_BIN/python3" ] && [ -x "$HOME/miniconda3/envs/base/bin/python3" ]; then
    PYTHON_ENV_BIN="$HOME/miniconda3/envs/base/bin"
fi

# Group size is total nEvents trained to NN
Group_Size=${Group_Size:-14000}

# Number of SLURM jobs
nJobs=${nJobs:-140}

# All energies assuming equal weight
Energies=(${Energies:-1 5 10 20 30 40 50 60 70 80 90 100 110 120})

# ----------------------------------------------
nEnergies=${#Energies[@]}


#Check to make sure Group size is divisible by job size
if (( Group_Size % nJobs != 0)); then
    echo "Group Size not divisible by number of jobs"
    exit 1
fi

Job_Size=$(( Group_Size / nJobs ))
echo "Job size: $Job_Size"

# Check to make sure nJobs is divisible by nEnergies
if (( nJobs % nEnergies != 0)); then
    echo "Number of jobs not divisible by number of energies"
    exit 1
fi

jobs_per_energy=$((nJobs / nEnergies))
echo "Jobs per energy: $jobs_per_energy"

gen_scripts(){

for sim_file in "$input_dir"/*.root; do
    if [ -n "${ONLY_ROOTS:-}" ]; then
        case " $ONLY_ROOTS " in
            *" $sim_file "*|*" $(basename "$sim_file") "*)
                ;;
            *)
                continue
                ;;
        esac
    fi

    fname=$(basename "$sim_file")
    energy=$(echo "$fname" | sed -n 's/.*_\([0-9][0-9.]*\)GeV_.*/\1/p')
    run=$(echo "$fname" | sed -n 's/.*GeV_\([0-9]\+\)_.*/\1/p')
    echo "Processing file $sim_file: energy $energy, run $run"

    local script_name="${OUTPUT_TAG}_simSPAD_${SPAD_Size}_${energy}_${run}.sh"

    cat << EOF > "${script_name}"
#!/bin/bash
#SBATCH -J "${OUTPUT_TAG}_simSPAD_${SPAD_Size}_${energy}_${run}"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p ${PARTITION}
#SBATCH -c 1
#SBATCH --mem-per-cpu=${MEMORY_PER_CPU}
#SBATCH --time=${SPAD_JOB_TIME_LIMIT}

set -euo pipefail

echo "Loading environment..."
export PATH="${PYTHON_ENV_BIN}:\$PATH"
echo "Environment loaded."
export TIME_SLICES="${TIME_SLICES}"

cd ${dest_dir}

failure_marker_dir="${dest_dir}/.failed_roots"
failure_marker="\${failure_marker_dir}/${fname}_SPAD${SPAD_Size}.failed"
root_key=\$(stat -c 'v2\t%n\t%s\t%Y' "${sim_file}" 2>/dev/null || true)
rm -f "\${failure_marker}"

set +e
timeout "${SIMSPADS_TIMEOUT_SECONDS}" python3 -u ${home_dir}/simSPADs.py \
"${sim_file}" \
"${energy}" \
${OUTPUT_TAG}_${energy}GeV_run${run}_SPAD${SPAD_Size}_CH${Channel_Size} \
${SPAD_Size} \
${Channel_Size}
simspads_rc=\$?
set -e

if [ "\${simspads_rc}" -ne 0 ]; then
    mkdir -p "\${failure_marker_dir}"
    {
        printf '%s\n' "\${root_key}"
        printf 'root=%s\n' "${sim_file}"
        printf 'exit_code=%s\n' "\${simspads_rc}"
        printf 'job_id=%s\n' "\${SLURM_JOB_ID:-unknown}"
        date '+failed_at=%F %T'
    } > "\${failure_marker}"
    echo "ERROR: simSPADs failed or timed out with exit code \${simspads_rc}; marker written to \${failure_marker}" >&2
    exit "\${simspads_rc}"
fi

rm -f "\${failure_marker}"

EOF
    chmod +x "${script_name}"
    sbatch "${script_name}"
done
}
gen_scripts