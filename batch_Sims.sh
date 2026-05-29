#!/bin/bash

home_dir=$PWD
sim_dir=${home_dir}/../DREAMSim/sim/build
dest_dir=/lustre/work/samumcki/pi_train

mkdir -p "${dest_dir}"
mkdir -p batch_jobs
mkdir -p batch_jobs/LOGDIR
cd batch_jobs

particle="pi+"

# Group size is total nEvents trained to NN
Group_Size=2000

# Number of SLURM jobs
nJobs=100

# All energies assuming equal weight
Energies=(1 5)
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

    gen_script() {
        local script_name="Simulations_${i}_${energy}.sh"

        cat << EOF > "$script_name"
#!/bin/bash
#SBATCH -J Simulations_${i}_${energy}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH --mem=8G

cd ${dest_dir}
export TMPDIR=/lustre/scratch/\$USER/tmp_\${SLURM_JOB_ID}
mkdir -p "\$TMPDIR"
trap 'rm -rf "\$TMPDIR"' EXIT


# Generate seeds
seed1=\$(( (RANDOM << 15) + RANDOM))
seed2=\$(( (RANDOM << 15) + RANDOM))

# Build temporary macro for seeds
echo "/random/setSeeds \$seed1 \$seed2" > random_${i}_${energy}.mac
cat ${sim_dir}/paramBatch03_single.mac >> random_${i}_${energy}.mac


singularity exec --cleanenv \
    --bind /lustre:/lustre \
    --bind "$TMPDIR":/tmp \
    /lustre/research/hep/yofeng/SimulationEnv/alma9forgeant4_sbox \
    bash --noprofile --norc -c "$sim_dir/exampleB4b -b random_${i}_${energy}.mac \
    -numberOfEvents ${Job_Size} -eventsInNtupe ${Job_Size} \
    -jobName sim_output_${Job_Size}events_${energy}GeV_${i}_${particle} \
    -gun_particle ${particle} -gun_energy_min ${energy} -gun_energy_max ${energy} \
    -sipmType 1"
sim_rc=\$?

if [ \$sim_rc -eq 0 ]; then
    echo "Simulation complete (seeds: \$seed1, \$seed2)"

else
    echo "ERROR: Simulation failed with exit code \$sim_rc"

fi

EOF

        sleep 0.5
        chmod +x "$script_name"
        sbatch "$script_name"

}

        gen_script

        echo "Simulation training ${i} for energy ${energy} initialized"

    done
done