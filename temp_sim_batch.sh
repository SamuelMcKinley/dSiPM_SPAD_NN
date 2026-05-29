#!/bin/bash

home_dir=$PWD
sim_dir=${home_dir}/../DREAMSim/sim/build
dest_dir=/lustre/work/samumcki/sim_results

mkdir -p "${dest_dir}"
mkdir -p batch_jobs
mkdir -p batch_jobs/LOGDIR
cd batch_jobs || exit 1

particle="pi+"

# Exact runs to remake
missing_runs=(5 7 10 11 12 33 34)
energy=1
Job_Size=20

for i in "${missing_runs[@]}"; do

    script_name="Simulations_${i}_${energy}.sh"

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

seed1=\$(( (RANDOM << 15) + RANDOM ))
seed2=\$(( (RANDOM << 15) + RANDOM ))

echo "/random/setSeeds \$seed1 \$seed2" > random_${i}_${energy}.mac
cat ${sim_dir}/paramBatch03_single.mac >> random_${i}_${energy}.mac

singularity exec --cleanenv \
    --bind /lustre:/lustre \
    --bind "\$TMPDIR":/tmp \
    /lustre/research/hep/yofeng/SimulationEnv/alma9forgeant4_sbox \
    bash --noprofile --norc -c "${sim_dir}/exampleB4b -b random_${i}_${energy}.mac \
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

    chmod +x "$script_name"
    sbatch "$script_name"
    echo "Simulation ${i} for ${energy} GeV initialized"

done