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

From `results/fmri_alpha_bold/alpha_bold_results.json`. Per-subject GLM with
occipital alpha envelope (8-12 Hz Hilbert, smooth 5 s, downsampled to TR,
z-scored, HRF-convolved with Glover) as the single parametric regressor;
motion + cosine drift as nuisance regressors; 6 mm smoothing. Threshold and
ROI stats restricted to each subject's stimloc-derived V1 mask.

| Subject | TRs | FD mean | V1 voxels | mean z[V1] | median z[V1] | % neg in V1 | z-map range |
|---|---:|---:|---:|---:|---:|---:|---|
| sub-500 | 360 | 0.118 mm | 197 | **-1.107** | -1.123 | **85.8 %** | [-5.05, 2.31] |
| sub-1070302 | 360 | 0.159 mm | 3559 | -0.078 | -0.050 | 52.4 % | [-4.79, 4.48] |

Interpretation:
- sub-500 shows the **canonical Goldman 2002 / Ingram 2024 negative alpha-BOLD
  coupling** in visual cortex — high alpha goes with low BOLD. 86 % of V1
  voxels are negative; the signal is unambiguous despite the small 197-voxel
  mask.
- sub-1070302 shows **essentially no alpha-BOLD coupling** in V1 (52 % negative
  = chance). Same subject that already showed EOEC reversal in both EEG and
  fMRI; rest now adds the third independent signal.

The three-signal cross-task pattern for sub-1070302:
1. EEG EOEC: occipital alpha reversed (EC < EO)
2. fMRI EOEC: visual-cortex BOLD reversed (EC > EO)
3. **Rest alpha-BOLD: coupling absent**

Most likely interpretation: drowsiness / sleep onset state during scanning
disrupts the normal alpha-vigilance / V1-engagement relationship. The
coupling test on the 6-min rest run is the highest-power version of the
effect (360 TRs vs 150 in EOEC).

Z-maps saved at `results/fmri_alpha_bold/sub-XXX/sub-XXX_alpha-bold-zmap.nii.gz`.
