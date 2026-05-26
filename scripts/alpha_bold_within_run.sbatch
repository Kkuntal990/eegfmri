#!/bin/bash
#SBATCH --job-name=alpha_bold_within
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/alpha_bold_within_%j.out
#SBATCH --error=logs/alpha_bold_within_%j.err
# ---------------------------------------------------------------
# Within-run sliding-window alpha-BOLD coupling dynamics.
# Tests whether sub-1070302's state-dependent coupling collapse
# is gradual (drift) or abrupt (state-switching), and whether
# window-level coupling tracks behaviour on the Sternberg runs.
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
STIMLOC_MASK_DIR="${STIMLOC_MASK_DIR:-/projects/bbnv/kkokate/eegfmri/results/stimloc_mask}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/alpha_bold_within_run}"

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

python scripts/alpha_bold_within_run.py \
    --bids-root "$BIDS_ROOT" \
    --stimloc-mask-dir "$STIMLOC_MASK_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Done: job $SLURM_JOB_ID"
