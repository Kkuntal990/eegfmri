#!/bin/bash
#SBATCH --job-name=hrf_lag_check
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hrf_lag_check_%j.out
#SBATCH --error=logs/hrf_lag_check_%j.err
# ---------------------------------------------------------------
# HRF-lag consistency check across rest, swm_run-1, swm_run-2.
# Cross-correlates alpha envelope vs V1-mean BOLD at lags -2..+12 TR
# to confirm fixed-Glover HRF alignment is not artifactually
# deflating cross-task reliability of alpha-BOLD coupling.
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
STIMLOC_MASK_DIR="${STIMLOC_MASK_DIR:-/projects/bbnv/kkokate/eegfmri/results/stimloc_mask}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/alpha_bold_hrf_lag}"

echo "========================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Host:             $(hostname)"
echo "BIDS root:        $BIDS_ROOT"
echo "Stimloc mask dir: $STIMLOC_MASK_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python scripts/hrf_lag_check.py \
    --bids-root "$BIDS_ROOT" \
    --stimloc-mask-dir "$STIMLOC_MASK_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Done: job $SLURM_JOB_ID"
