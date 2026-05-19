# EEG-fMRI project — Claude operating manual

This file is the durable context for working on this repo with Claude Code.
Read this first whenever you resume work.

## Project overview

Simultaneous EEG-fMRI dataset (UCLA, Siemens Prisma 3T, 128-ch EGI HydroCel,
TR=1s, multiband-4). Pipeline currently has:

- EEG side: BIDS loader (`eegfmri_loader.py`) → MR + BCG artifact removal →
  resample 250 Hz → braindecode `BaseConcatDataset`. Validated on EOEC task:
  outside-scanner Berger effect is textbook; inside-scanner CSP+LDA 75-88 %.
- fMRI side: **in progress.** Currently building a within-subject EO vs EC
  classifier as the analogue of the EEG validation.

Subjects currently processed: `sub-500`, `sub-1070302`. Scale-out target up
to n=156 (UCLA Dataset 1) + n=67 (Dataset 2 with face tasks) when the
pipeline solidifies.

## Repos and sync

Three copies stay in lockstep through GitHub:

```
LOCAL  (~/Desktop/eegfmri)                    ─┐
                                                ├─ origin: github.com/Kkuntal990/eegfmri (main hub)
DELTA  (/projects/bbnv/kkokate/eegfmri)       ─┘
```

**Workflow:** edit code on local Mac → commit → push → `git pull` on Delta.
Never edit code on Delta unless fixing a bug discovered there.

Heavy outputs (logs, results, NIfTI, EEGLAB sets, checkpoints) stay out of
git — see `.gitignore`.

## Data paths

| Location | Path |
|---|---|
| BIDS dataset on Delta | `/work/hdd/bbnv/kuntal/eegfmri_data` |
| Repo on Delta | `/projects/bbnv/kkokate/eegfmri` |
| Results on Delta | `/projects/bbnv/kkokate/eegfmri/results` |
| Logs on Delta | `/projects/bbnv/kkokate/eegfmri/logs` |
| Local BIDS copy (sub-500, sub-1070302 only) | `~/Desktop/eegfmri/sub-*` |

## Delta access — operating rules

- SSH alias: `ssh delta` (preferred — see ~/.ssh/config)
- **Never run code on the login node.** Anything that takes more than ~1 minute
  (env setup, preprocessing, training, validation, model fits, even
  `pip install`) **must** be submitted via `sbatch`.
- Short read-only checks (`ls`, `cat`, `git status`, `squeue`) are OK on login.
- Account: `bbnv-delta-gpu` (our only allocation — verified via `accounts`)
- GPU partition: `gpuA40x4`. **Every sbatch must include `--gpus-per-task=1`**
  even for CPU-only workloads, because the GPU account refuses jobs that
  request zero GPUs.
- Log convention: `logs/<jobname>_%j.out` and `logs/<jobname>_%j.err`

## Environment

Conda env on both sides: **`eegfmri`** (Python 3.12).

Currently installed (verified):
```
mne 1.11.0   mne-bids 0.18.0   braindecode 1.3.2
numpy 2.4.3  scipy 1.17.1      scikit-learn 1.8.0
pandas 3.0.1 matplotlib 3.10.8 joblib  tqdm
torch 2.10.0 torchaudio 2.10.0 rotary-embedding-torch
```

fMRI additions (install via `scripts/setup_fmri_env.sbatch`):
```
nilearn  nibabel
```

Module-load incantation used in every sbatch:
```bash
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eegfmri
```

## Existing sbatch scripts (templates to copy from)

| Script | Purpose |
|---|---|
| `scripts/preprocess_eegfmri.sbatch` | EEG MR+BCG artifact removal, resample, braindecode dataset build |
| `scripts/train_eoec.sbatch` | EEG EO/EC classification (ShallowFBCSPNet, EEGNetv4) |
| `scripts/validate_eeg.sbatch` | EEG alpha-power Berger + CSP/SVM validation |
| `scripts/setup_fmri_env.sbatch` | One-shot pip install of `nilearn` + `nibabel` + `antspyx` |
| `scripts/fmri_stimloc_mask.sbatch` | Per-subject visual-cortex functional localizer from the stimloc task; produces masks under `results/stimloc_mask/sub-XXX/` |
| `scripts/fmri_eoec.sbatch` | fMRI EO/EC classification (within-subject). Auto-picks-up `results/stimloc_mask/` for a real V1 ROI classifier. |

When creating a new sbatch, copy one of these as a template. Keep the
common header (account, partition, log paths, module + conda block) consistent.

## Software stack — what's available on Delta

| | Status |
|---|---|
| Native FSL / ANTs / FreeSurfer / AFNI modules | ❌ none |
| Apptainer (system-wide) | ✅ `/usr/bin/apptainer` 1.4.2 |
| Miniforge / conda | ✅ via `miniforge3-python` module |
| CUDA toolkit | ✅ `cudatoolkit/25.3_11.8` |
| pytorch-conda 2.8 module | ✅ |

Implication: for serious fMRI preprocessing (motion correction, MNI
normalisation, BBR coreg), we will eventually pull the fMRIPrep Apptainer
image. Until then, within-subject native-space analysis via nilearn + nibabel
covers the smoke test.

## Results so far

EEG side (validation_results.json at `/projects/bbnv/kkokate/eegfmri/results/validation`):

| Recording | Occipital α EC/EO | p | Cohen's d | CSP+LDA | SVM band-power |
|---|---|---|---|---|---|
| sub-500 ses-01 (outside) | 3.10× | 2e-5 | 0.94 | 90.5 % | 92.6 % |
| sub-500 ses-02 (inside) | 0.99× | 0.12 | 0.26 | 88.2 % | 66.6 % |
| sub-1070302 ses-02 (inside) | 0.85× ↓ | 0.71 | -0.12 | 75.1 % | 61.8 % |

Cross-subject deep models (`results/eoec/`): ~50 % (chance) — expected at n=2.

fMRI side: pending the in-progress EO/EC analysis.

## House style for new code

- Pure functions; no globals.
- Use `argparse`; defaults point at Delta paths.
- Log to stdout; sbatch captures it.
- Save numeric results to a JSON file under `results/<experiment>/`.
- Heuristic for heavy I/O: load once, cache to disk if same data is needed
  more than twice.
- Prefer `nilearn` + `nibabel` for fMRI; prefer `mne` + `braindecode` for EEG.
- HRF lag for fMRI labelling: shift labels +5 TRs (TR=1s).
