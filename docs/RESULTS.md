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

## Cross-state reliability sprint (Gap-#4) — N=2 preliminary evidence

Four analyses targeted at the question: *is voxelwise alpha-BOLD coupling
within native V1 a reliable within-subject biomarker across cognitive states?*
All run on rest + swm_run-1 + swm_run-2, the three within-session
alpha-rhythm-bearing tasks.

### Per-task alpha-BOLD GLM

From `results/alpha_bold_per_task/per_task_results.json`. Same GLM as the
rest experiment, generalised to swm_run-1 and swm_run-2.

| Subject | Task | TRs | FD | V1 vox | **mean z[V1]** | % neg V1 |
|---|---|---:|---:|---:|---:|---:|
| sub-500 | rest | 360 | 0.117 | 197 | **−1.093** | 85.3 % |
| sub-500 | swm_run-1 | 513 | 0.113 | 196 | **−0.473** | 67.9 % |
| sub-500 | swm_run-2 | 503 | 0.116 | 192 | **−0.885** | 84.4 % |
| sub-1070302 | rest | 360 | 0.159 | 3559 | −0.129 | 55.2 % |
| sub-1070302 | swm_run-1 | 563 | 0.102 | 3346 | −0.341 | 69.1 % |
| sub-1070302 | swm_run-2 | 509 | 0.100 | 3382 | −0.063 | 51.9 % |

sub-500 stays solidly canonical (negative, > 67 % negative voxels) across
all three cognitive states. sub-1070302 oscillates near zero, with two of
three states essentially at chance.

### Voxelwise ICC(3,1) within V1

From `results/alpha_bold_reliability/group_summary.json`. Shrout & Fleiss
ICC(3,1), pairwise Pearson r, and |z| > 2.3 Dice across the three z-maps,
restricted to each subject's stimloc V1 mask.

| Subject | V1 vox | **ICC(3,1)** | r rest~swm1 | r rest~swm2 | r swm1~swm2 | dice swm1~swm2 |
|---|---:|---:|---:|---:|---:|---:|
| sub-500 | 197 | **+0.011** | −0.181 | +0.129 | +0.074 | 0.29 |
| sub-1070302 | 3575 | **−0.013** | −0.098 | −0.073 | +0.122 | 0.06 |

Voxelwise reliability is essentially zero in both subjects, even in the
canonical-coupling subject whose ROI-mean signal is robust. The same V1
voxels do not preferentially negative-couple across runs.

### HRF-lag consistency

From `results/alpha_bold_hrf_lag/summary.csv`. Cross-correlated the
occipital alpha envelope vs. V1-mean BOLD at lags −2..+12 TR per
(subject, task); report lag of most-negative r.

| Subject | Task | **Lag (TR)** | r at lag | Verdict |
|---|---|---:|---:|---|
| sub-500 | rest | 4 | **−0.115** | canonical (Glover peak) |
| sub-500 | swm_run-1 | 6 | **−0.094** | canonical |
| sub-500 | swm_run-2 | 4 | **−0.106** | canonical |
| sub-1070302 | rest | 7 | **−0.116** | canonical-but-late |
| sub-1070302 | swm_run-1 | 7 | −0.013 | no coupling |
| sub-1070302 | swm_run-2 | 3 | −0.032 | no coupling |

sub-500's HRF lag is stable at 4–6 TR (Glover-canonical) across all three
tasks — the fixed-Glover GLM is well-aligned. Voxelwise ICC ≈ 0 is therefore
a *real reliability finding*, not an HRF-misalignment artefact.

sub-1070302's coupling exists at rest (r=−0.116 at lag 7) but disappears
under cognitive load (|r| < 0.05). Coupling is **state-dependent**, not
absent.

### Within-run sliding-window dynamics

From `results/alpha_bold_within_run/summary.csv`. Window = 50 TR, step
= 25 TR, lag = +5 TR (Glover canonical). For swm runs, window r is
correlated with mean trial-level reactionTime and accuracy.

| Subject | Task | #win | r mean | r SD | r min | r max | % neg | r~RT | r~ACC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-500 | rest | 13 | −0.063 | 0.273 | −0.550 | +0.339 | 53.8 % | — | — |
| sub-500 | swm_run-1 | 19 | −0.051 | 0.234 | −0.591 | +0.507 | 68.4 % | −0.488 | +0.174 |
| sub-500 | swm_run-2 | 18 | −0.120 | 0.306 | −0.560 | +0.511 | 66.7 % | −0.061 | **−0.414** |
| sub-1070302 | rest | 13 | −0.118 | 0.241 | −0.460 | +0.370 | 69.2 % | — | — |
| sub-1070302 | swm_run-1 | 21 | −0.090 | 0.234 | −0.452 | +0.339 | 61.9 % | +0.196 | −0.012 |
| sub-1070302 | swm_run-2 | 19 | −0.036 | 0.218 | −0.411 | +0.466 | 68.4 % | −0.293 | +0.326 |

Within-run coupling is highly non-stationary in both subjects (Allen 2018
directly observable: SD ≈ 0.22–0.31, individual windows swing −0.59 ↔
+0.51). Behavioural validity correlations are mixed across the 4 swm
rows — one canonical hit (sub-500 swm_run-2 ACC = −0.41), one anti-canonical
RT hit, two nulls. Underpowered at N=2 × 18–21 windows; deferred to n=156.

### Methodological verdict (triangulated across all 4 analyses)

| Spatial scale | Reliable across states? | Evidence |
|---|---|---|
| Voxelwise within V1 | **No** | ICC ≈ 0 in both subjects |
| Within-run window-mean r | **No** | SD ~ 0.25; every task swings through zero |
| **Whole-run ROI-mean r at canonical lag** | **Yes (in canonical subjects)** | sub-500: −0.10 ± 0.01 across all 3 states |
| Across-run sign-consistency of whole-run r | **The phenotype** | sub-500 stays negative; sub-1070302 collapses on swm |

This is the methodological conclusion the n=156 proposal builds on:
voxelwise reliability is a non-starter, but the **ROI-mean cross-state
phenotype** is operationally measurable in 2/2 subjects and stratifies
canonical-vs-vigilance-vulnerable in the predicted direction.
