#!/bin/bash
#SBATCH --job-name=eeg_validate
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate_eeg_%j.out
#SBATCH --error=logs/validate_eeg_%j.err
# ---------------------------------------------------------------
# Validate EEG data quality: alpha power analysis + CSP/SVM classification
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/validation}"

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "BIDS root:  $BIDS_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python scripts/validate_eeg.py \
    --bids-root "$BIDS_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --window-size 500

echo "Done: job $SLURM_JOB_ID"
