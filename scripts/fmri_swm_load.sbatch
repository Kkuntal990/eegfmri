#!/bin/bash
#SBATCH --job-name=fmri_swm_load
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fmri_swm_load_%j.out
#SBATCH --error=logs/fmri_swm_load_%j.err
# ---------------------------------------------------------------
# Sternberg WM load (load1 vs load5) classification, within-subject,
# leave-one-run-out CV across swm_run-1 / swm_run-2. CPU-only nilearn
# + sklearn; GPU requested only because our account demands it.
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/fmri_swm_load}"

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "BIDS root:  $BIDS_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python scripts/fmri_swm_load.py \
    --bids-root "$BIDS_ROOT" \
    --output-dir "$OUTPUT_DIR"

echo "Done: job $SLURM_JOB_ID"
