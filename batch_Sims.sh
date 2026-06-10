#!/bin/bash

#
#   This code is used to batch GEANT4 simulations onto the HPCC
#   assuming DREAMSim repository next to dSiPM_SPAD_NN directory
#   and on commit a2b7a91
#



home_dir=$PWD
sim_dir=${SIM_DIR:-${home_dir}/../DREAMSim/sim/build}
dest_dir=${DEST_DIR:-/lustre/work/$USER/pi_train}

mkdir -p "${dest_dir}"
mkdir -p batch_jobs
mkdir -p batch_jobs/LOGDIR
cd batch_jobs

# --------- Adjustable parameters -------------
particle=${PARTICLE:-pi+}

# Group size is total nEvents trained to NN
Group_Size=${Group_Size:-2000}

# Number of SLURM jobs
nJobs=${nJobs:-100}

# All energies assuming equal weight
Energies=(${Energies:-1 5})
PARTITION=${PARTITION:-nocona}
MEMORY=${MEMORY:-16G}
# The old 8G setting can pack too many Singularity/ROOT startups on one node.
# If an already-running controller passes 8G, lift it for newly generated jobs.
if [ "$MEMORY" = "8G" ]; then
    MEMORY=16G
fi
SIM_STARTUP_JITTER_SECONDS=${SIM_STARTUP_JITTER_SECONDS:-120}
SIM_ATTEMPTS=${SIM_ATTEMPTS:-3}
SIM_RETRY_SLEEP_SECONDS=${SIM_RETRY_SLEEP_SECONDS:-120}
ONLY_JOBS=${ONLY_JOBS:-}
SIM_MACRO=${SIM_MACRO:-${sim_dir}/paramBatch03_single.mac}
SINGULARITY_IMAGE=${SINGULARITY_IMAGE:-/lustre/research/hep/yofeng/SimulationEnv/alma9forgeant4_sbox}

# ---------------------------------------------
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


for energy in "${Energies[@]}"; do

    for (( i=0; i<jobs_per_energy; i++ )); do

        if [ -n "$ONLY_JOBS" ]; then
            wanted=0
            for job_spec in $ONLY_JOBS; do
                if [ "$job_spec" = "${energy}:${i}" ]; then
                    wanted=1
                    break
                fi
            done
            if [ "$wanted" -ne 1 ]; then
                continue
            fi
        fi

    gen_script() {
        local script_name="Simulations_${i}_${energy}.sh"

        cat << EOF > "$script_name"
#!/bin/bash
#SBATCH -J Simulations_${i}_${energy}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p ${PARTITION}
#SBATCH --mem=${MEMORY}

set -euo pipefail

cd ${dest_dir}
export TMPDIR=/lustre/scratch/\$USER/tmp_\${SLURM_JOB_ID}
mkdir -p "\$TMPDIR" "\$TMPDIR/home" "\$TMPDIR/xdg-cache" "\$TMPDIR/singularity-cache" "\$TMPDIR/singularity-tmp"
export SINGULARITY_CACHEDIR="\$TMPDIR/singularity-cache"
export SINGULARITY_TMPDIR="\$TMPDIR/singularity-tmp"
ulimit -c 0 || true
trap 'rm -rf "\$TMPDIR"' EXIT

SIM_STARTUP_JITTER_SECONDS=${SIM_STARTUP_JITTER_SECONDS}
SIM_RETRY_SLEEP_SECONDS=${SIM_RETRY_SLEEP_SECONDS}
if [ "\$SIM_STARTUP_JITTER_SECONDS" -gt 0 ]; then
    startup_sleep=\$(( RANDOM % (SIM_STARTUP_JITTER_SECONDS + 1) ))
    echo "Startup jitter: sleeping \${startup_sleep}s before launching Singularity"
    sleep "\$startup_sleep"
fi

# Generate seeds
seed1=\$(( (RANDOM << 15) + RANDOM))
seed2=\$(( (RANDOM << 15) + RANDOM))

# Build temporary macro for seeds. Drop commands unsupported by this build.
{
    echo "/random/setSeeds \$seed1 \$seed2"
    grep -v '^[[:space:]]*/physics_list/list[[:space:]]*$' ${SIM_MACRO}
} > random_${i}_${energy}.mac


sim_rc=1
attempt=1
while [ "\$attempt" -le ${SIM_ATTEMPTS} ]; do
    echo "Simulation attempt \$attempt/${SIM_ATTEMPTS}"
    if [ "\$attempt" -gt 1 ]; then
        rm -f mc_sim_output_${Job_Size}events_${energy}GeV_${i}_${particle}*.root
        retry_sleep=\$(( SIM_RETRY_SLEEP_SECONDS + (RANDOM % (SIM_RETRY_SLEEP_SECONDS + 1)) ))
        echo "Retry sleep: \${retry_sleep}s"
        sleep "\$retry_sleep"
    fi

    set +e
    singularity exec --cleanenv \
        --bind /lustre:/lustre \
        --bind "\$TMPDIR":/tmp \
        ${SINGULARITY_IMAGE} \
        bash --noprofile --norc -c "export HOME=/tmp/home XDG_CACHE_HOME=/tmp/xdg-cache ROOT_HIST=0; $sim_dir/exampleB4b -b random_${i}_${energy}.mac \
        -numberOfEvents ${Job_Size} -eventsInNtupe ${Job_Size} \
        -jobName sim_output_${Job_Size}events_${energy}GeV_${i}_${particle} \
        -gun_particle ${particle} -gun_energy_min ${energy} -gun_energy_max ${energy} \
        -sipmType 1"
    sim_rc=\$?
    set -e

    if [ "\$sim_rc" -eq 0 ]; then
        echo "Simulation complete on attempt \$attempt (seeds: \$seed1, \$seed2)"
        exit 0
    fi

    echo "WARNING: Simulation attempt \$attempt failed with exit code \$sim_rc" >&2
    attempt=\$((attempt + 1))
done

echo "ERROR: Simulation failed after ${SIM_ATTEMPTS} attempts; final exit code \$sim_rc" >&2
rm -f mc_sim_output_${Job_Size}events_${energy}GeV_${i}_${particle}*.root
exit "\$sim_rc"

EOF

        sleep 0.5
        chmod +x "$script_name"
        sbatch "$script_name"

}

        gen_script

        echo "Simulation training ${i} for energy ${energy} initialized"

    done
done