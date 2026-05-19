#!/bin/bash
#SBATCH --job-name=fmri_env_setup
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/fmri_env_setup_%j.out
#SBATCH --error=logs/fmri_env_setup_%j.err
# ---------------------------------------------------------------
# One-shot: add nilearn + nibabel to the eegfmri conda env.
# Re-running is safe (pip is idempotent).
# ---------------------------------------------------------------

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Host:   $(hostname)"

cd /projects/bbnv/kkokate/eegfmri

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri

python -V
pip install --upgrade pip
pip install nilearn nibabel

echo "--- verifying imports ---"
python -c "import nilearn, nibabel; print('nilearn', nilearn.__version__); print('nibabel', nibabel.__version__)"

echo "Done: job $SLURM_JOB_ID"
