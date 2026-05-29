#!/bin/bash

home_dir=$PWD
sim_dir=${home_dir}/../DREAMSim/sim/build
dest_dir=/lustre/work/samumcki/SPAD_results
input_dir=/home/samumcki/sim_results/train_sims

mkdir -p "${dest_dir}"
mkdir -p batch_jobs
mkdir -p batch_jobs/LOGDIR
cd batch_jobs

particle="pi+"

SPAD_Size=200x200
Channel_Size=1000x1000

# Group size is total nEvents trained to NN
Group_Size=10000

# Number of SLURM jobs
nJobs=100

# All energies assuming equal weight
Energies=(10 20 30 40 50 60 70 80 90 100)
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

    fname=$(basename "$sim_file")
    energy=$(echo "$fname" | sed -n 's/.*_\([0-9]\+\)GeV_.*/\1/p')
    run=$(echo "$fname" | sed -n 's/.*GeV_\([0-9]\+\)_pi+.*/\1/p')
    echo "Processing file $sim_file: energy $energy, run $run"

    local script_name="simSPAD_${energy}_${run}.sh"

    cat << EOF > "simSPAD_${energy}_${run}.sh"
#!/bin/bash
#SBATCH -J "simSPAD_${energy}_${run}"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 1
#SBATCH --mem-per-cpu=16G

echo "Loading environment..."
export PATH=~/miniconda3/envs/base/bin:\$PATH
echo "Environment loaded."

cd ${dest_dir}

python3 -u ${home_dir}/Mickey_Sim.py \
"${sim_file}" \
"${energy}" \
M_SPAD_output_${energy}GeV_run${run}_SPAD${SPAD_Size}_Channel${Channel_Size} \
${SPAD_Size} \
${Channel_Size}

EOF
    sleep 0.5
    chmod +x "${script_name}"
    sbatch "${script_name}"
done
}
gen_scripts