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
PYTHON_ENV_BIN=${PYTHON_ENV_BIN:-$HOME/miniconda3/envs/dsipm-spad/bin}
if [ ! -x "$PYTHON_ENV_BIN/python3" ]; then
    if [ -x "$HOME/miniconda3/envs/base/bin/python3" ]; then
        PYTHON_ENV_BIN="$HOME/miniconda3/envs/base/bin"
    elif [ -x "$HOME/miniconda3/bin/python3" ]; then
        PYTHON_ENV_BIN="$HOME/miniconda3/bin"
    fi
fi

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
export PATH="${PYTHON_ENV_BIN}:\$PATH"
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
        set +e
        python3 - <<'PYROOTCHECK'
import glob
import os
import sys
import ROOT

ROOT.gROOT.SetBatch(True)
expected = int("${Job_Size}")
pattern = "mc_sim_output_${Job_Size}events_${energy}GeV_${i}_${particle}*.root"
paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
if not paths:
    print(f"ERROR: No ROOT output matched {pattern}", file=sys.stderr)
    raise SystemExit(2)
path = paths[0]
f = ROOT.TFile(path, "READ")
if not f or f.IsZombie():
    print(f"ERROR: ROOT output is zombie/unreadable: {path}", file=sys.stderr)
    raise SystemExit(2)
tree = f.Get("tree")
if not tree:
    print(f"ERROR: ROOT output missing tree: {path}", file=sys.stderr)
    f.Close()
    raise SystemExit(2)
entries = int(tree.GetEntries())
if entries != expected:
    print(f"ERROR: ROOT output has {entries}/{expected} entries: {path}", file=sys.stderr)
    f.Close()
    raise SystemExit(2)
if tree.GetEntry(0) <= 0:
    print(f"ERROR: Could not read first ROOT entry: {path}", file=sys.stderr)
    f.Close()
    raise SystemExit(2)
st = os.stat(path)
key = f"v3\t{os.path.abspath(path)}\t{st.st_size}\t{int(st.st_mtime)}"
marker_dir = ".validated_roots"
os.makedirs(marker_dir, exist_ok=True)
marker = os.path.join(marker_dir, os.path.basename(path) + ".ok")
with open(marker, "w", encoding="utf-8") as handle:
    handle.write(key + "\n")
    handle.write(f"root={os.path.abspath(path)}\n")
    handle.write(f"expected_events={expected}\n")
    handle.write(f"entries={entries}\n")
f.Close()
print(f"Validated ROOT output: {path} ({entries}/{expected} entries)")
PYROOTCHECK
        validate_rc=\$?
        set -e
        if [ "\$validate_rc" -eq 0 ]; then
            echo "Simulation complete on attempt \$attempt (seeds: \$seed1, \$seed2)"
            exit 0
        fi
        sim_rc=\$validate_rc
        echo "WARNING: Simulation attempt \$attempt produced invalid ROOT output; validation exit code \$sim_rc" >&2
        rm -f mc_sim_output_${Job_Size}events_${energy}GeV_${i}_${particle}*.root
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