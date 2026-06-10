# dSiPM SPAD Neural Network Workflow

This project runs a full detector-simulation and neural-network workflow:

1. Generate particle events with DREAMSim/Geant4.
2. Convert the simulated photons into SPAD time-sliced tensors.
3. Make photon-analysis summaries and plots.
4. Train one neural network for each SPAD size.
5. Run predictions on a separate event set.

The default particle is `pi+`, but the workflow can use other DREAMSim-supported particles by changing `PARTICLE` in `run_streamlined_workflow.sh`.

Most commands in this README are meant to be copied and pasted.

## Where This Runs

This workflow is only intended to run on the HPCC/SLURM cluster.

## Environment

If you do not already have a Python environment for this project, create one with conda on HPCC:

```bash
conda create -n dsipm-spad python=3.11 -y
conda activate dsipm-spad
conda install -c conda-forge root numpy matplotlib scipy scikit-learn pytorch torchvision -y
```

Check that the important imports work:

```bash
python3 -c "import ROOT, numpy, matplotlib, torch; print('environment ok')"
```

The batch scripts currently add this path automatically:

```bash
~/miniconda3/envs/base/bin
```

If you use a different environment name, either activate it before running or update the `export PATH=...` lines in the batch scripts to point at your environment. Geant4 simulation itself runs inside the configured Singularity/Apptainer image, so most users should not need to install Geant4 manually.

## Folder Layout

The two repositories should sit next to each other like this:

```text
some_parent_folder/
  dSiPM_SPAD_NN/
  DREAMSim/
```

From inside `dSiPM_SPAD_NN`, the DREAMSim repo should be reachable as:

```text
cd ../DREAMSim
```

## Required DREAMSim Version

DREAMSim is here:

```text
https://github.com/TTU-HEP/DREAMSim
```

This workflow is tied to this exact commit:

```text
a2b7a91f48985a0962ceef6ef2527050c07d143e
```

Direct commit link:

```text
https://github.com/TTU-HEP/DREAMSim/commit/a2b7a91f48985a0962ceef6ef2527050c07d143e
```

To check the DREAMSim version:

```bash
cd ../DREAMSim
git rev-parse HEAD
```

If it prints a different commit, switch to the required one:

```bash
git checkout a2b7a91f48985a0962ceef6ef2527050c07d143e
```

Then return to this repo:

```bash
cd ../dSiPM_SPAD_NN
```

## Before You Run

Make sure you are in this repository:

```bash
pwd
```

The path should end with:

```text
dSiPM_SPAD_NN
```

Check that the main scripts are present:

```bash
ls run_streamlined_workflow.sh batch_Sims.sh batch_simSPADs.sh
```

Check that DREAMSim was built:

```bash
ls ../DREAMSim/sim/build/exampleB4b
```

If that file is missing, DREAMSim needs to be built before this workflow can run.

## Run the Full Workflow

Submit the full workflow with:

```bash
sbatch run_streamlined_workflow.sh
```

The workflow will submit many smaller SLURM jobs in stages. It waits for simulations, then SPAD tensor jobs, then photon analysis, then neural-network training and prediction.

## Check Whether It Is Running

To see your current jobs:

```bash
squeue -u $USER
```

The main controller job will be named something like:

```text
run_stre
```

Simulation jobs will be named something like:

```text
Simulations_...
```

SPAD jobs will include:

```text
simSPAD
```

Neural-network jobs will include:

```text
NN_train
NN_pred
```

## Check the Main Log

When you submit with `sbatch`, SLURM prints a job number, for example:

```text
Submitted batch job 12345678
```

The main log will be:

```text
slurm-12345678.out
```

To watch the newest messages:

```bash
tail -n 80 slurm-12345678.out
```

Replace `12345678` with your actual job number.

## Default Run Size

The default production run currently uses:

```text
pi+ particles
14,000 training events
14,000 prediction events
140 Geant4 jobs for training
140 Geant4 jobs for prediction
SPAD sizes: 1x1, 5x5, 10x10, 20x20, 50x50, 100x100
32 time slices
```

Training and prediction use separate event sets.

## Change Simple Settings

Most settings are near the top of:

```text
run_streamlined_workflow.sh
```

Common settings:

```bash
PARTICLE=pi+
ENERGIES="1 5 10 20 30 40 50 60 70 80 90 100 110 120"
SPAD_SIZES="1x1 5x5 10x10 20x20 50x50 100x100"
TRAIN_SIM_GROUP_SIZE=14000
PREDICT_SIM_GROUP_SIZE=14000
```

For a tiny test run, set:

```bash
QUICK_TEST=1
```

For the normal large run, use:

```bash
QUICK_TEST=0
```

## Where Outputs Go

Large outputs are not saved in Git. By default they are written under your own cluster work directory:

```text
/lustre/work/$USER/pi_train
/lustre/work/$USER/pi_predict
/lustre/work/$USER/SPAD_results
```

Important output folders:

```text
Training ROOT files:     /lustre/work/$USER/pi_train
Prediction ROOT files:   /lustre/work/$USER/pi_predict
SPAD tensors:            /lustre/work/$USER/SPAD_results/train_<SPAD_SIZE>
Prediction tensors:      /lustre/work/$USER/SPAD_results/predict_<SPAD_SIZE>
Photon analysis:         /lustre/work/$USER/SPAD_results/photon_full_analysis
Cumulative npy tensors:  /lustre/work/$USER/SPAD_results/cumulative_npy
NN models/results:       NN_Analysis/<SPAD_SIZE>_model
```

## If a Run Stops or Fails

The workflow is designed to reuse good files.

If you run it again, it checks which ROOT files and tensors already exist. Good files are reused. Missing or bad files are regenerated.

So the usual recovery step is simply:

```bash
sbatch run_streamlined_workflow.sh
```

Do not manually delete large output folders unless you are sure you want to start over.

Files ending in `.bad_YYYYMMDD_HHMMSS` are old or bad files.

## Helpful Commands

Show jobs:

```bash
squeue -u $USER
```

Show recent main-log messages:

```bash
tail -n 80 slurm-JOBID.out
```

Count training ROOT files:

```bash
find /lustre/work/$USER/pi_train -maxdepth 1 -name '*100events*.root' | wc -l
```

Count prediction ROOT files:

```bash
find /lustre/work/$USER/pi_predict -maxdepth 1 -name '*100events*.root' | wc -l
```

Check Git status before committing:

```bash
git status --short
```
