#!/bin/bash
#SBATCH --job-name=vigilance_wm
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vigilance_wm_%j.out
#SBATCH --error=logs/vigilance_wm_%j.err
# ---------------------------------------------------------------
# Trial-locked vigilance x WM analysis on the Sternberg runs.
# Computes per-trial alpha (baseline/encode/engaged), frontal-midline
# theta during maintenance, and HRF-lagged V1 BOLD, then runs five
# sub-analyses: behaviour, alpha desync, theta maintenance,
# load-locked alpha-BOLD coupling, vigilance x RT correlation.
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
STIMLOC_MASK_DIR="${STIMLOC_MASK_DIR:-/projects/bbnv/kkokate/eegfmri/results/stimloc_mask}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/vigilance_wm}"

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

python scripts/vigilance_wm_trials.py \
    --bids-root "$BIDS_ROOT" \
    --stimloc-mask-dir "$STIMLOC_MASK_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Done: job $SLURM_JOB_ID"
