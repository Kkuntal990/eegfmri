#!/bin/bash
#SBATCH --job-name=alpha_bold_pertask
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/alpha_bold_pertask_%j.out
#SBATCH --error=logs/alpha_bold_pertask_%j.err
# ---------------------------------------------------------------
# Week-1 Gap-#4 deliverable: per-task alpha-BOLD GLM for
# {rest, swm_run-1, swm_run-2} on each subject. CPU-only nilearn
# GLM + EEG preprocessing; --gpus-per-task=1 required by account.
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/alpha_bold_per_task}"
SUBJECTS="${SUBJECTS:-500 1070302}"
TASKS="${TASKS:-rest swm_run-1 swm_run-2}"

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "BIDS root:  $BIDS_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Subjects:   $SUBJECTS"
echo "Tasks:      $TASKS"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python scripts/fmri_alpha_bold_per_task.py \
    --bids-root "$BIDS_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --subjects $SUBJECTS \
    --tasks $TASKS

echo "Done: job $SLURM_JOB_ID"
