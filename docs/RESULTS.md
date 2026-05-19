# Results — detailed numbers

Authoritative log of what each experiment produced. Updated as new jobs land.

## EEG EOEC validation

From `results/validation/validation_results.json`.

| Recording | Occipital α EC/EO | p | Cohen's d | CSP+LDA | SVM band-power |
|---|---|---|---|---|---|
| sub-500 ses-01 (outside scanner) | 3.10× | 2e-5 | 0.94 | 90.5 % | 92.6 % |
| sub-500 ses-02 (inside scanner) | 0.99× | 0.12 | 0.26 | 88.2 % | 66.6 % |
| sub-1070302 ses-02 (inside scanner) | **0.85× ↓ reversed** | 0.71 | -0.12 | 75.1 % | 61.8 % |

Cross-subject deep models in `results/eoec/`: ~50 % (chance) — expected at n=2.

## fMRI EOEC classification

From `results/fmri_eoec/fmri_eoec_results.json`. Within-subject, paired-block CV
(3 folds, each fold = 1 EC + 1 EO block), ANTs motion correction, FD + FD′ + FD
as nuisance regressors.

| Subject | PCA50+LinearSVC | LinearSVC voxels | PosteriorMean+LogReg | VisualCortexMean+LogReg |
|---|---:|---:|---:|---:|
| sub-500 | 92.0 % | 91.3 % | 50.0 % | 42.0 % |
| sub-1070302 | 89.3 % | 90.0 % | 35.3 % | **44.0 % (reversed)** |

Motion (FD mean): sub-500 0.098 mm, sub-1070302 0.111 mm. Both well under the
1 mm scrub threshold.

## Stimloc functional V1 localizer

From `results/stimloc_mask/stimloc_mask_results.json`. Nilearn FirstLevelModel,
HRF=glover, 6 mm smoothing, threshold z > 2.3 with top-1% fallback if fewer
than 100 voxels pass.

| Subject | Stim trials | Brain voxels | Visual voxels | z_max | Notes |
|---|---:|---:|---:|---:|---|
| sub-500 | 32 | 85,531 | 194 | 3.55 | Weak stimloc activation; mask is small but real |
| sub-1070302 | 32 | 86,318 | 3547 | 6.20 | Clean visual-cortex map |

Schema quirk: sub-500's events.tsv has 12 columns with duration in milliseconds;
sub-1070302's has 8 columns with duration in seconds. Parser auto-detects by
median magnitude.

## Cross-modal observation — sub-1070302

Same subject, reversed direction in both modalities at the same time:

| Modality | Expected (canonical) | sub-1070302 observed |
|---|---|---|
| EEG occipital alpha | EC > EO (Berger) | **EC < EO (ratio 0.85)** |
| fMRI visual cortex BOLD | EO > EC | **EC > EO** |

Most likely: drowsiness / sleep onset during EC blocks. This becomes the
positive control for the alpha-BOLD coupling experiment.

## fMRI alpha-BOLD GLM (rest task)

Pending — `scripts/fmri_eeg_alpha_glm.sbatch`. Expected: negative z inside the
V1 mask (canonical coupling — high alpha = low BOLD) for sub-500;
attenuated / positive z for sub-1070302 (matches drowsiness hypothesis).

Output will go to `results/fmri_alpha_bold/alpha_bold_results.json`.
