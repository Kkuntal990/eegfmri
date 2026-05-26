# Proposal — Cross-state reliability of ROI-mean alpha-BOLD coupling as a vigilance-instability phenotype

**Requesting access to:** the n=156 simultaneous EEG-fMRI cohort (Dataset 1)
and the n=67 face-task extension (Dataset 2).

**Three-sentence summary.** We have built and validated a per-subject
alpha-BOLD coupling pipeline on the two pilot subjects (sub-500, sub-1070302)
and demonstrated, in pre-registered terms, that *voxelwise* alpha-BOLD
coupling within native V1 is **not** within-subject reliable across cognitive
states — but the **ROI-mean coupling slope at the canonical Glover lag is
reliable in the canonical-coupling subject and collapses, state-dependently,
in the vigilance-vulnerable subject**. With the n=156 cohort we would
establish the across-subject distribution of this cross-state ROI-mean
phenotype and test whether it (i) stratifies subjects into canonical vs
vigilance-vulnerable clusters and (ii) tracks an independent vigilance
covariate vector (EEG, EOG, BOLD-template) and Sternberg working-memory
behaviour.

---

## 1 The question

Goldman et al. 2002 established that occipital alpha power is *negatively*
coupled to V1 BOLD at the group level. Two decades of follow-up work
(Laufs 2003; Allen 2018; Sadaghiani/Wirsich 2025; Jiricek 2026) has refined
the **group-level map** but has never measured **within-subject reliability**
of the coupling at the voxel or ROI scale, and has not asked whether this
reliability — independent of magnitude — is itself a biomarker for vigilance
instability.

This matters because: (a) reliability is the prerequisite for any biomarker
use of alpha-BOLD coupling; (b) Elliott et al. 2020 (the "reliability
fallacy") has changed task-fMRI norms but has not been extended to
multimodal coupling; (c) vigilance is currently regressed out or treated as
an exclusion criterion in EEG-fMRI, when it may be the most clinically
informative axis the modality offers.

## 2 Literature gap — verified novelty (2022–2026)

A 2022–2026 systematic check of Google Scholar / PubMed / bioRxiv on
"alpha-BOLD" × "test-retest" × "reliability" × "within-subject" finds
**no paper** that fits voxelwise or ROI alpha-BOLD coupling per subject
per session and reports formal reliability (ICC / Pearson r / Dice).
The closest precedents:

| Paper | What they did | What is left open |
|---|---|---|
| **Nakuci, Muldoon et al. 2023** (*Sci Reports*) | 8-session within-subject ICC of *unimodal* connectomes (EEG **or** fMRI separately) | Cross-modal coupling-slope ICC — a different quantity |
| **Xavier, Sadaghiani, Wirsich 2025** (*Imaging Neurosci*) | *Group-level* consistency of RSN-band-power correlations across datasets | Within-subject reliability — the Elliott-2020 distinction |
| **Allen et al. 2018** (*NeuroImage*) | Within-session non-stationarity of alpha-BOLD coupling (sliding window) | Between-session systematic variance is a distinct construct; nobody has averaged over the non-stationarity to test trait-level reliability |
| **Goldman 2002 / Jiricek 2026 cell-type gradient preprint** | Group-mean alpha-BOLD maps; cortical cell-type gradient | Within-subject voxelwise/ROI test-retest — the missing reliability step |

Our novelty claim, narrowed to what the literature actually leaves open:

> First within-subject, cross-cognitive-state reliability of voxelwise and
> ROI-mean alpha-BOLD coupling in native space, with reliability
> (operationalised as cross-state sign-consistency at canonical HRF lag)
> proposed as a vigilance-instability biomarker rather than a nuisance
> covariate.

## 3 Preliminary evidence (N=2, this lab)

We ran four pre-registered analyses on the two pilot subjects across three
within-session alpha-rhythm-bearing tasks (rest, 6 min; swm_run-1, ~9 min;
swm_run-2, ~9 min). All native-space, all cached, all reproducible from
`scripts/` in our public repo.

### 3.1 Per-task alpha-BOLD GLM (Glover HRF, 6 mm smoothing)

| Subject | rest mean z[V1] | swm-1 mean z[V1] | swm-2 mean z[V1] | rest %neg | swm-1 %neg | swm-2 %neg |
|---|---:|---:|---:|---:|---:|---:|
| **sub-500** (canonical) | **−1.093** | **−0.473** | **−0.885** | 85.3 % | 67.9 % | 84.4 % |
| **sub-1070302** (vigilance-vulnerable) | −0.129 | −0.341 | −0.063 | 55.2 % | 69.1 % | 51.9 % |

### 3.2 Voxelwise ICC(3,1) across the three z-maps in V1

| Subject | V1 vox | ICC(3,1) | r rest~swm1 | r rest~swm2 | r swm1~swm2 |
|---|---:|---:|---:|---:|---:|
| sub-500 | 197 | **+0.011** | −0.18 | +0.13 | +0.07 |
| sub-1070302 | 3575 | **−0.013** | −0.10 | −0.07 | +0.12 |

Voxelwise alpha-BOLD coupling is essentially unreliable across states in
both subjects — even in the canonical-coupling subject whose ROI-mean
signal is robust.

### 3.3 HRF-lag consistency (cross-correlation, lags −2..+12 TR)

| Subject | Task | lag of min r (TR) | r at lag | Verdict |
|---|---|---:|---:|---|
| sub-500 | rest / swm-1 / swm-2 | 4 / 6 / 4 | −0.115 / −0.094 / −0.106 | **Glover-canonical, stable** |
| sub-1070302 | rest | 7 | −0.116 | canonical-but-late |
| sub-1070302 | swm-1 / swm-2 | (n/a) | −0.013 / −0.032 | **state-dependent collapse** |

This rules out HRF-misalignment as a cause of the voxelwise-ICC≈0 result —
in the canonical subject the fixed Glover HRF is well-aligned across all
three states.

### 3.4 Within-run sliding-window dynamics

50-TR windows, step 25 TR, lag = +5 TR. Window-level r SD ≈ 0.22–0.31 in
both subjects (Allen 2018 reproduced). Window-level behavioural-validity
correlations (r ↔ reactionTime / accuracy) are mixed at N=2 × 18–21
windows — one canonical hit, one anti-canonical hit, two nulls — exactly
the noise floor expected at this scale. **The validity check therefore
requires scale to test.**

### 3.5 Methodological verdict

| Spatial scale | Reliable across states? |
|---|---|
| Voxelwise within V1 | **No** (ICC ≈ 0 in both subjects) |
| Within-run window-mean r | **No** (SD ~ 0.25; every task swings through zero) |
| **Whole-run ROI-mean r at canonical lag** | **Yes** in canonical subjects (sub-500: −0.10 ± 0.01 across 3 states) |
| Across-run sign-consistency of whole-run r | **The phenotype** (canonical vs state-dependent-collapse) |

The N=2 evidence supports advancing to scale with the **ROI-mean cross-state
sign-consistency** as the operational biomarker, not the voxelwise map.

## 4 Hypotheses for n=156 — pre-registered

**H1** (existence). The across-subject distribution of cross-state ROI-mean
r in native-V1 has bimodal or heavy-tailed structure, with one mode at the
canonical Goldman-2002 value (≈ −0.10 to −0.15) and a second mode near
zero, rather than a narrow Gaussian around the group mean.

**H2** (clustering). HDBSCAN or two-mixture Gaussian clustering on
(mean_r_across_3_states, cross_state_SD_of_r) recovers ≥ 2 clusters with
> 70 % stability under leave-one-task-out resampling.

**H3** (convergent validity). The cluster assignment is predicted, at α =
0.01, by an independent 5-element vigilance covariate vector:
 1. EEG occipital alpha EO/EC ratio (Berger; Barry 2007)
 2. EEG theta/alpha ratio at posterior channels (Hori 1994; N1 marker)
 3. EOG slow-eye-movement count
 4. Tagliazucchi 2012 EEG vigilance index
 5. Falahpour 2018 BOLD vigilance-template correlation
With a hold-out 30 % of subjects for validation.

**H4** (predictive validity). The canonical-cluster subjects have lower
within-subject reactionTime SD and higher accuracy on swm load5 trials
than the vigilance-vulnerable cluster (mixed-effects regression, subject
ID as random effect), pre-registered direction.

**H5** (methodological correction). Voxelwise ICC(3,1) within V1 has a
group mean ≤ 0.10 (Noble 2019 fMRI-FC mean is 0.29; our prediction is that
*cross-modal coupling* maps are *less* reliable than unimodal FC maps).
This is the headline methodological contribution: a stop-sign on voxelwise
alpha-BOLD coupling reports.

## 5 Analysis plan — concrete and locked

All scripts already exist in the public repo (`scripts/`) and have run
end-to-end on the 2 pilot subjects in ≤ 12 min total per subject after the
cache warms.

1. **Pipeline per subject** (∼ 12 min each, cached):
 - EEG MR + BCG removal (eegfmri_loader)
 - ANTs rigid-body motion correction
 - Per-subject stimloc V1 mask (z > 2.3 with top-1 % fallback)
 - Per-task alpha-BOLD GLM (Glover HRF, 6 mm smoothing, motion + cosine drift nuisance)
 - HRF-lag check (cross-correlation −2..+12 TR)
 - Whole-run ROI-mean r at canonical lag (+5 TR)
 - 50-TR sliding-window dynamics + per-window behaviour

2. **Group analyses** (∼ 4 h on a single Delta GPU node):
 - Cluster (mean_r, SD_r) plane → H1, H2
 - Bayesian mixed-effects regression of cluster ↔ vigilance vector → H3
 - Pooled window-level r ↔ RT/ACC across ∼ 3 000 swm windows → H4
 - Voxelwise ICC(3,1) group distribution → H5

3. **Pre-registration**: hypotheses + analysis plan locked on OSF before
 any data beyond the 2 pilot subjects is touched. We will publish the
 pre-registration ID with the proposal acceptance.

4. **Reproducibility**: every output is one Slurm job from a fresh checkout.
 We will share the cache + outputs with the collaborator before paper
 submission.

## 6 Risks and pre-emptions

| Risk | How we will defend |
|---|---|
| **"Nakuci 2023 did this."** | Replicate their unimodal connectome-ICC on the n=156 cohort as a positive control alongside our cross-modal coupling-slope ICC. The two quantities are distinct; the side-by-side replication makes that explicit. |
| **"Allen 2018 already showed alpha-BOLD is non-stationary."** | Frame within-session non-stationarity (Allen) and between-session reliability (us) as distinct constructs. Pre-register the test that subjects with low within-session α-power variance also have high between-state ROI ICC. |
| **"HRF varies across states → low ICC is artefactual."** | Refit GLM with (a) Glover canonical, (b) glover + temporal + dispersion derivatives, (c) FIR-estimated subject-specific HRF from stimloc. Report ICC under all three. The N=2 lag check already shows the canonical subject's HRF is stable. |
| **"6 mm smoothing inflates spatial autocorrelation → ICC is undercounted."** | Sensitivity sweep at 0 mm / 3 mm / 6 mm on the full cohort, included in supplementary. |
| **"You're fishing for clusters."** | Pre-registered cluster method (HDBSCAN, default params, no parameter tuning post-hoc). Stability assessed by leave-one-task-out and 30 % held-out validation. |
| **"Drowsiness is the same as motion."** | Vigilance covariate vector is constructed from EEG, EOG, and BOLD-template indicators that are independent of FD. We will additionally report group results on a low-motion subset (FD < 0.2 mm). |

## 7 What we are asking for, and what the collaborator gets

**Asking for:**

1. Read access to the BIDS root containing the n=156 simultaneous EEG-fMRI
 cohort, identical structure to sub-500 / sub-1070302.
2. (If available) the n=67 face-task extension dataset, structured the same
 way.
3. Demographics + any vigilance / questionnaire variables already collected
 (ASRS attention, MEWS, Karolinska Sleepiness Scale, etc.).
4. One ~30-minute kickoff meeting to align on the pre-registered hypotheses
 before OSF lock.

**The collaborator gets:**

1. Co-authorship on the resulting paper (anticipated *Imaging Neuroscience*
 or *NeuroImage* methodology track).
2. The fully reproducible pipeline (cached EEG cleanup, BOLD motion
 correction, per-subject V1 masks, GLM, reliability, vigilance vector)
 returned as a code drop.
3. The per-subject reliability + vigilance-cluster assignments as a
 derivative dataset they can reuse in their own future work.
4. Replication of Nakuci 2023's unimodal connectome ICC on their cohort
 as a side-product positive control.

## 8 Timeline (post-access)

| Week | Deliverable |
|---|---|
| 1 | Pipeline deployed end-to-end on first 10 subjects; QC pass / fail report |
| 2 | Full n=156 cache warmed; preliminary group-level (mean_r, SD_r) plane |
| 3 | Vigilance covariate vector built; H1 + H2 tested |
| 4 | H3 + H5 tested; methodological-correction figure |
| 5 | H4 (window-level behavioural validity, pooled) |
| 6 | Draft manuscript circulated to all co-authors |

Total: 6 weeks of analyst time. Compute is bbnv-delta-gpu (already
funded); no additional resources requested.

## 9 Contact

Kuntal Kokate — kukokate@ucsd.edu
Repo: github.com/Kkuntal990/eegfmri (full pipeline + N=2 evidence reproducible)
Pre-registration draft: available on request; will be OSF-locked before data access begins.
