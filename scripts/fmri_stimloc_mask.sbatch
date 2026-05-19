#!/bin/bash
#SBATCH --job-name=fmri_stimloc_mask
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fmri_stimloc_mask_%j.out
#SBATCH --error=logs/fmri_stimloc_mask_%j.err
# ---------------------------------------------------------------
# Per-subject visual-cortex functional localizer from the stimloc task.
# Produces a thresholded mask per subject under
# results/stimloc_mask/sub-XXX/.
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/stimloc_mask}"

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

python scripts/fmri_stimloc_mask.py \
    --bids-root "$BIDS_ROOT" \
    --output-dir "$OUTPUT_DIR"

echo "Done: job $SLURM_JOB_ID"
