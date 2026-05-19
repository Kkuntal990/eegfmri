#!/bin/bash
#SBATCH --job-name=eoec_train
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eoec_train_%j.out
#SBATCH --error=logs/eoec_train_%j.err
# ---------------------------------------------------------------
# EOEC (Eyes Open vs Closed) classification with EEGConformer
#
# Usage:
#   cd /projects/bbnv/kkokate/eegfmri
#   sbatch scripts/train_eoec.sbatch
# ---------------------------------------------------------------

set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/work/hdd/bbnv/kuntal/eegfmri_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bbnv/kkokate/eegfmri/results/eoec}"

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "BIDS root:  $BIDS_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "GPU:        $SLURM_GPUS_ON_NODE"
echo "========================================"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

nvidia-smi

# Run ShallowFBCSPNet
python scripts/train_eoec.py \
    --bids-root "$BIDS_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model ShallowFBCSPNet \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --patience 20 \
    --device cuda

# Run EEGNetv4
python scripts/train_eoec.py \
    --bids-root "$BIDS_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model EEGNetv4 \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --patience 20 \
    --device cuda

echo "Done: job $SLURM_JOB_ID"
