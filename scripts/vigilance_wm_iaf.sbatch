#!/bin/bash
#SBATCH --job-name=vigilance_iaf
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vigilance_iaf_%j.out
#SBATCH --error=logs/vigilance_iaf_%j.err
# ---------------------------------------------------------------
# IAF sensitivity test for Analysis 4 (Risk C from reviewer
# feedback): compute per-subject IAF from rest task, filter EEG
# at IAF +/- 2 Hz instead of 8-12 Hz, re-run the Bridwell lag sweep.
# Side-by-side comparison with the fixed-band result.
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
STIMLOC_MASK_DIR="${STIMLOC_MASK_DIR:-/projects/bbnv/kkokate/eegfmri/results/stimloc_mask}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/vigilance_wm_iaf}"
FIXED_LAG_SWEEP="${FIXED_LAG_SWEEP:-/projects/bbnv/kkokate/eegfmri/results/vigilance_wm/lag_sweep.csv}"

echo "========================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Host:             $(hostname)"
echo "BIDS root:        $BIDS_ROOT"
echo "Output dir:       $OUTPUT_DIR"
echo "Fixed lag sweep:  $FIXED_LAG_SWEEP"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python scripts/vigilance_wm_iaf.py \
    --bids-root "$BIDS_ROOT" \
    --stimloc-mask-dir "$STIMLOC_MASK_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --fixed-lag-sweep "$FIXED_LAG_SWEEP"

echo "Done: job $SLURM_JOB_ID"
