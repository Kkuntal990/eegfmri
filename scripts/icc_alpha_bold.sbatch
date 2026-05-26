#!/bin/bash
#SBATCH --job-name=icc_alpha_bold
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/icc_alpha_bold_%j.out
#SBATCH --error=logs/icc_alpha_bold_%j.err
# ---------------------------------------------------------------
# Week-2 deliverable: ICC(3,1) + Pearson + Dice across the three
# per-task alpha-BOLD coupling z-maps, inside each subject's
# stimloc-derived V1 mask.
# ---------------------------------------------------------------

set -euo pipefail

ZMAP_DIR="${ZMAP_DIR:-/projects/bbnv/kkokate/eegfmri/results/alpha_bold_per_task}"
STIMLOC_MASK_DIR="${STIMLOC_MASK_DIR:-/projects/bbnv/kkokate/eegfmri/results/stimloc_mask}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/alpha_bold_reliability}"

echo "========================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Host:             $(hostname)"
echo "Z-map dir:        $ZMAP_DIR"
echo "Stimloc mask dir: $STIMLOC_MASK_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python scripts/icc_alpha_bold.py \
    --zmap-dir "$ZMAP_DIR" \
    --stimloc-mask-dir "$STIMLOC_MASK_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Done: job $SLURM_JOB_ID"
