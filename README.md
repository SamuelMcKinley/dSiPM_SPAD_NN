# dSiPM SPAD NN Workflow

End-to-end workflow for generating DREAMSim particle samples, converting them into time-sliced dSiPM/SPAD tensors, aggregating photon statistics, training one neural network per SPAD size, running predictions on a separate event set, and plotting the outputs. The particle species is configurable with `PARTICLE` and defaults to `pi+`.

## Required Companion Repo

This repository expects a sibling DREAMSim checkout at:

```text
../DREAMSim
```

The workflow has been tested against this exact DREAMSim commit:

```text
a2b7a91f48985a0962ceef6ef2527050c07d143e
```

The executable and macro are expected by default at:

```text
../DREAMSim/sim/build/exampleB4b
../DREAMSim/sim/build/paramBatch03_single.mac
```

You can override these with `SIM_DIR` and `SIM_MACRO`. On HPCC the default Singularity image is:

```text
/lustre/research/hep/yofeng/SimulationEnv/alma9forgeant4_sbox
```

Override it with `SINGULARITY_IMAGE` if needed.

## Main Run

Submit the complete workflow from the repo root:

```bash
sbatch run_streamlined_workflow.sh
```

The script uses `$USER` for working paths by default, for example:

```text
/lustre/work/$USER/pi_train
/lustre/work/$USER/pi_predict
/lustre/work/$USER/SPAD_results
```

Training and prediction simulations are separate event sets. Existing valid ROOT files are reused; missing or bad chunks are regenerated. The default particle is `pi+`, but other DREAMSim-supported particles can be selected with `PARTICLE`.

## Default Production Settings

The current default production run is configured for:

```text
14,000 train events
14,000 prediction events
140 Geant4 jobs per sample
SPAD sizes: 1x1 5x5 10x10 20x20 50x50 100x100
32 time slices
```

Important adjustable variables are near the top of `run_streamlined_workflow.sh`, including `PARTICLE`, `ENERGIES`, `SPAD_SIZES`, `TIME_SLICES`, `TRAIN_SIM_GROUP_SIZE`, `PREDICT_SIM_GROUP_SIZE`, and stage switches like `RUN_GEANT4` or `RUN_NN_TRAIN`.

## Outputs

Large generated outputs are intentionally ignored by Git. The workflow writes them mostly under `/lustre/work/$USER`, including:

```text
SPAD tensors:       /lustre/work/$USER/SPAD_results/train_<SPAD_SIZE> and predict_<SPAD_SIZE>
Photon analysis:    /lustre/work/$USER/SPAD_results/photon_full_analysis
Cumulative tensors: /lustre/work/$USER/SPAD_results/cumulative_npy
NN outputs:         NN_Analysis/<SPAD_SIZE>_model/
SLURM logs:         slurm-*.out, workflow_logs/, batch_jobs/LOGDIR/
```

## Pre-Push Checks

Before committing code changes, run:

```bash
bash -n run_streamlined_workflow.sh
bash -n batch_Sims.sh
bash -n batch_simSPADs.sh
python3 -m py_compile simSPADs.py aggregate_photon_stats.py plot.py NN_Analysis/model_copy/dataset.py NN_Analysis/model_copy/train.py
git status --short
```

Make sure no generated ROOT, tensor, checkpoint, plot, or SLURM output files are staged.
