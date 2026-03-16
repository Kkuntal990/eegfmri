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
# Usage:
#   cd /projects/bbnv/kkokate/eegfmri
#   sbatch scripts/preprocess_eegfmri.sbatch
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "BIDS root:  $BIDS_ROOT"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python scripts/run_preprocess.py --bids-root "$BIDS_ROOT"

echo "Done: job $SLURM_JOB_ID"
