# EEG-fMRI project — Claude operating manual

Durable context for this repo. Read on resume.

Detailed docs in `docs/`:
- `docs/RESULTS.md` — full numerical results across experiments

## Project overview

Simultaneous EEG-fMRI (UCLA, Siemens Prisma 3T, 128-ch EGI HydroCel, TR=1s,
multiband-4). Subjects processed so far: `sub-500`, `sub-1070302`. Scale-out to
n=156 (Dataset 1) + n=67 (Dataset 2, face tasks) when the pipeline solidifies.

Status:
- **EEG**: BIDS loader → MR + BCG artifact removal → 250 Hz → braindecode. EOEC
  validated; outside-scanner Berger effect textbook; inside-scanner classifier
  75–88 %.
- **fMRI**: nilearn + nibabel + antspyx pipeline. EOEC + stimloc V1 localizer
  done. Current experiment: alpha-BOLD coupling GLM on the rest task.
- **Key finding**: sub-1070302 shows reversed EOEC direction in *both* EEG and
  fMRI — interpreted as drowsiness/sleep onset during EC blocks. Becomes the
  positive control for the alpha-BOLD experiment.

## Repos and sync

```
LOCAL  (~/Desktop/eegfmri)              ─┐
                                          ├─ origin: github.com/Kkuntal990/eegfmri
DELTA  (/projects/bbnv/kkokate/eegfmri) ─┘
```

Workflow: edit on Mac → commit → push → `git pull` on Delta. Never edit code on
Delta unless fixing a bug discovered there. Heavy outputs (`logs/`, `results/`,
`*.nii.gz`, `*.set`, `*.fdt`, `*.pt`) stay out of git — see `.gitignore`.

## Data paths

| Location | Path |
|---|---|
| BIDS root on Delta | `/work/hdd/bbnv/kuntal/eegfmri_data` |
| Repo on Delta | `/projects/bbnv/kkokate/eegfmri` |
| Results on Delta | `/projects/bbnv/kkokate/eegfmri/results` |
| Logs on Delta | `/projects/bbnv/kkokate/eegfmri/logs` |
| Local BIDS copy (sub-500, sub-1070302 only) | `~/Desktop/eegfmri/sub-*` |

## Delta access — operating rules

- SSH alias: `ssh delta`.
- **Never run code on the login node.** Anything taking more than ~1 min (env
  setup, preprocessing, training, model fits, even `pip install`) **must** go
  via `sbatch`. Short read-only checks (`ls`, `cat`, `git status`, `squeue`)
  are fine on login.
- Account: `bbnv-delta-gpu` (our only allocation; verified via `accounts`).
- GPU partition: `gpuA40x4`. **Every sbatch must include `--gpus-per-task=1`**
  even for CPU-only jobs — the GPU account refuses zero-GPU jobs.
- Log convention: `logs/<jobname>_%j.out` and `logs/<jobname>_%j.err`.

## Environment

Conda env `eegfmri` (Python 3.11) on both sides.

Currently installed (verified): mne 1.11, mne-bids 0.18, braindecode 1.3.2,
torch 2.10, numpy 2.3, scipy 1.15, scikit-learn 1.8, pandas 3.0, nilearn 0.13,
nibabel 5.4, antspyx 0.6.3, plus matplotlib, joblib, tqdm.

Module-load incantation used in every sbatch:
```bash
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri
```

## sbatch scripts (templates — copy when adding new ones)

| Script | Purpose |
|---|---|
| `setup_fmri_env.sbatch` | One-shot pip install (nilearn, nibabel, antspyx) |
| `preprocess_eegfmri.sbatch` | EEG MR+BCG removal, resample, braindecode dataset |
| `train_eoec.sbatch` | EEG EO/EC deep classifiers (ShallowFBCSPNet, EEGNetv4) |
| `validate_eeg.sbatch` | EEG alpha-power Berger + CSP/SVM validation |
| `fmri_stimloc_mask.sbatch` | Per-subject functional V1 localizer (stimloc) |
| `fmri_eoec.sbatch` | fMRI EO/EC within-subject classifier; uses stimloc V1 |
| `fmri_eeg_alpha_glm.sbatch` | Alpha-BOLD coupling GLM on rest task |
| `fmri_swm_load.sbatch` | Sternberg WM load1-vs-load5 within-subject classifier (cross-modal, leave-one-run-out) |
| `scripts/_cache.py` | Disk-cache helpers for cleaned EEG (.fif) and motion-corrected BOLD (.nii.gz + .npy) under `derivatives/cache/`. Every fMRI script uses these so reruns skip the ~12 min of MR+BCG+motion preprocessing. Delete `derivatives/cache/` to force a rebuild. |

## Software stack on Delta

| | Status |
|---|---|
| Native FSL / ANTs / FreeSurfer / AFNI modules | ❌ none |
| Apptainer (system) | ✅ `/usr/bin/apptainer` 1.4.2 |
| Miniforge / conda | ✅ via `miniforge3-python` |
| CUDA toolkit | ✅ `cudatoolkit/25.3_11.8` |
| pytorch-conda 2.8 module | ✅ |

For serious fMRI preprocessing (MNI normalisation, BBR coreg, FreeSurfer
surfaces), we'll eventually pull the fMRIPrep Apptainer image. Until then,
within-subject native-space analysis via nilearn + nibabel + antspyx covers
the smoke tests and current experiments.

## Tasks available in BIDS root

| Task | EEG ses-01 | EEG ses-02 | fMRI ses-02 | Duration | Note |
|---|---|---|---|---|---|
| eoec | ✓ | ✓ | ✓ ~3 min | 6× 30 s alternating EC/EO blocks | done |
| rest | – | ✓ | ✓ ~6 min | eyes-open resting | alpha-BOLD experiment uses this |
| stimloc | ✓ | ✓ | ✓ ~3.5 min | passive dot stimuli, count red crosses | mask built |
| swm_run-1 | ✓ | ✓ | ✓ ~8.5 min | Sternberg spatial WM, load 1 vs 5 | unused |
| swm_run-2 | ✓ | ✓ | ✓ ~8.4 min | same as run-1 | unused |
| cpt | ✓ (sub-500 only) | – | – | letter Go/No-Go | EEG-only |

## House style for new code

- Pure functions, no globals.
- `argparse`; defaults point at Delta paths.
- Log to stdout; sbatch captures it.
- Save numeric outputs to JSON under `results/<experiment>/`.
- Prefer `nilearn` + `nibabel` + `antspyx` for fMRI; `mne` + `braindecode` for EEG.
- HRF lag for fMRI block labelling: shift labels +5 TRs (TR=1 s).
- Use `nilearn.glm.first_level.compute_regressor` with `'glover'` for explicit
  HRF convolution.
- Whenever a new fMRI script needs visual cortex: load the per-subject
  stimloc mask from `results/stimloc_mask/sub-XXX/`.
- Append new results to `docs/RESULTS.md` rather than growing this file.
