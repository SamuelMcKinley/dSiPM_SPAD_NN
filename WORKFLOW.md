# Workflow Notes

Use `run_streamlined_workflow.sh` as the main entry point. It submits separate Geant4 batches for training and prediction, reuses each batch across every configured SPAD size, waits between SLURM phases, trains one NN per SPAD size, predicts on separate prediction tensors, writes cumulative `.npy` tensors, aggregates photon statistics, and produces plots.

## DREAMSim Requirement

The sibling `../DREAMSim` repository must be on commit:

```text
a2b7a91f48985a0962ceef6ef2527050c07d143e
```

Default paths are:

```text
SIM_DIR=../DREAMSim/sim/build
SIM_MACRO=../DREAMSim/sim/build/paramBatch03_single.mac
SINGULARITY_IMAGE=/lustre/research/hep/yofeng/SimulationEnv/alma9forgeant4_sbox
```

Override those environment variables if your checkout or image path differs.

## Run Everything

```bash
sbatch run_streamlined_workflow.sh
```

The runner has editable variables at the top. Paths are user-generic and default to `/lustre/work/$USER/...`. Training and prediction ROOT files are kept separate so the NN is not trained and evaluated on the same event set.

## Individual Pieces

Run Geant4 simulations only:

```bash
./batch_Sims.sh
```

Convert ROOT files to SPAD tensors:

```bash
SPAD_Size=20x20 OUTPUT_TAG=train ./batch_simSPADs.sh
SPAD_Size=20x20 INPUT_DIR=/lustre/work/$USER/pi_predict OUTPUT_TAG=predict ./batch_simSPADs.sh
```

Train from a copied NN template:

```bash
cd NN_Analysis/20x20_model
SPAD_SIZE=20x20 TENSOR_DIR=/lustre/work/$USER/SPAD_results/train_20x20 ./trainNN.sh
```

Predict from the same copied model directory:

```bash
SPAD_SIZE=20x20 TENSOR_DIR=/lustre/work/$USER/SPAD_results/predict_20x20 ./predictNN.sh
```

Plot outputs:

```bash
python3 plot.py /lustre/work/$USER/SPAD_results/photon_full_analysis \
  --nn-root NN_Analysis \
  --tensor-root /lustre/work/$USER/SPAD_results/predict_20x20
```
