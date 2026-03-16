#!/bin/bash
#SBATCH --job-name=eegfmri_preprocess
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
# ---------------------------------------------------------------
# EEG-fMRI Preprocessing on Delta (NCSA)
#
# Loads raw EEG-fMRI data, applies MR gradient + BCG artifact
# removal (ses-02), standard preprocessing, and verifies the
# braindecode pipeline end-to-end.
#
# Usage:
#   cd /projects/bbnv/kkokate/eegfmri
#   sbatch scripts/preprocess_eegfmri.sbatch
#
# Override data path:
#   sbatch --export=ALL,BIDS_ROOT=/other/path scripts/preprocess_eegfmri.sbatch
# ---------------------------------------------------------------

set -euo pipefail

# --- Configuration ---
BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "BIDS root:  $BIDS_ROOT"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

# Load conda and activate environment
module reset
module load miniforge3-python
eval "\$(conda shell.bash hook)"
conda activate eegfmri

# Run preprocessing
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from eegfmri_loader import load_dataset

bids_root = '${BIDS_ROOT}'

# Load all recordings with full preprocessing pipeline
print('Loading all recordings with preprocessing...')
dataset = load_dataset(
    bids_root,
    preprocess=True,
    resample_freq=250.0,
    pick_eeg_only=True,
    mr_artifact_removal=True,
    bcg_artifact_removal=True,
    n_jobs=1,
)

print(f'\nLoaded {len(dataset.datasets)} recordings')
print(dataset.description)

for i, ds in enumerate(dataset.datasets):
    raw = ds.raw
    print(f'  [{i}] sub-{ds.description["subject"]} ses-{ds.description["session"]} '
          f'task-{ds.description["task"]} run-{ds.description["run"]}: '
          f'{len(raw.ch_names)} ch, {raw.info["sfreq"]} Hz, {raw.times[-1]:.1f}s')

# Windowing test
from braindecode.preprocessing import create_fixed_length_windows
windows = create_fixed_length_windows(
    dataset, window_size_samples=500, window_stride_samples=250, drop_last_window=True
)
print(f'\nTotal windows: {len(windows)}')
X, y, info = windows[0]
print(f'Window shape: {X.shape}')
print('\nAll checks passed!')
"

echo "Done: job $SLURM_JOB_ID"
