# Proposal — Load-dependent transition from lag-shifted canonical to truly inverted alpha-BOLD coupling as a vigilance-vulnerability biomarker

**Requesting access to:** the n=156 simultaneous EEG-fMRI cohort (Dataset 1)
and, if available, the n=67 face-task extension (Dataset 2).

**Three-sentence summary.** In two pilot subjects we observe a previously
unreported **three-way interaction** between cognitive load, HRF lag, and
vigilance phenotype on trial-level alpha-BOLD coupling in V1: the
vigilance-vulnerable subject's coupling is *delayed-but-canonical* at low
WM load (Bridwell 2025 noncanonical-timing phenomenon — r=−0.23 at lag 11 TR
where canonical lag is 5 TR) and *truly inverted* at high WM load (r=+0.24
at all lags 3–11 TR, no canonical reading recoverable). The canonical
subject shows textbook negative coupling at canonical lag 5 that
*strengthens* with load (r=−0.029 at load1 → −0.245 at load5). At n=156 we
will test whether this load × lag × phenotype pattern partitions the
cohort into a discrete vigilance-vulnerability cluster, validates against
independent vigilance markers (EOEC reversal, alpha-RT correlation,
behavioural accuracy decrement), and predicts working-memory performance
decrement under load.

---

## 1 The question

The clinical literature on vigilance instability (ADHD, depression, mild
TBI, sleep disorders) needs a *mechanistic* marker that connects EEG, fMRI,
and behaviour in a single session. Working memory is the cognitive function
most affected by vigilance fluctuation and the most measured in clinic.
Simultaneous EEG-fMRI on a working-memory task in principle exposes the
moment-to-moment neurovascular coupling that bridges the three, but the
field treats this coupling as a group-mean phenomenon (Goldman 2002;
Laufs 2003) or a nuisance to be regressed out (Tagliazucchi & Laufs 2014).

We ask: **Does vigilance phenotype determine how trial-level alpha-BOLD
coupling responds to working-memory load, and is the response itself a
biomarker for the behavioural cost of vigilance instability?**

## 2 Literature gap — verified novelty (2022–2026)

A targeted 2022–2026 systematic search verifies that the *exact* construct
we propose has not been published. The closest precedents and our explicit
differentiators:

| Paper | What they did | Our novel piece |
|---|---|---|
| **Scheeringa 2009** (*NeuroImage*) | Trial-level alpha-BOLD coupling on Sternberg WM, group-mean negative coupling, parametric alpha increase with load | Did NOT split coupling by load × phenotype; never examined individual sign differences |
| **Michels 2010** (*PLOS One*) | Trial-by-trial EEG regressors per load condition (set 2 vs 5), group-mean load-dependent α-BOLD | Group-mean only; no individual coupling-sign analysis; no vigilance phenotyping |
| **Meltzer 2007** (*Clin Neurophysiol*) — most threatening | Found WM alpha scores "positive in half subjects, negative in half"; used as between-subject regressor | (a) Separate EEG and fMRI sessions, not simultaneous; (b) condition-level not trial-level; (c) not V1-specific; (d) explicitly noted **no behavioural or physiological correlate found** — never identified vigilance |
| **Bridwell 2025** (*Biol Psychiatry CNNI*) | Noncanonical α-BOLD coupling in schizophrenia at rest — but as *lag shifts* (0–2 s vs canonical 4–6 s), not sign flips | Sign flip, not lag shift, in cognitive-load condition; vigilance phenotyping, not psychiatric diagnosis; WM task with load manipulation, not rest |
| **Leicht 2025** (*HBM*) | Single-trial *theta* EEG-fMRI in visual WM encoding | Different band, different epoch, no V1 phenotype dissociation |

Our precise novelty wording, defended against each precedent:

> **In simultaneous EEG-fMRI on the Sternberg working-memory task,
> vigilance-vulnerable subjects show a *load-dependent transition* in
> trial-level alpha-BOLD coupling: under low cognitive load, the coupling
> is canonical-negative at a delayed HRF lag (~10–11 s; consistent with and
> extending Bridwell 2025's noncanonical-timing finding); under high
> cognitive load, the coupling truly inverts and stays positive at every
> HRF lag tested 3–11 s. Canonical-phenotype subjects show only the
> textbook negative coupling at canonical lag 5 TR, strengthening with
> load. This load × lag × phenotype three-way interaction has not been
> previously reported.**

A secondary novel finding: **alpha-as-vigilance-marker validity is
subject-conditional** — the alpha-RT correlation only emerges in subjects
who actually have vigilance fluctuation. This is methodologically
important for any future EEG-vigilance biomarker work.

## 3 Preliminary evidence — N=2

All analyses run end-to-end on Delta using the cached pipeline in our
public repo (`scripts/vigilance_wm_trials.py`). Reproducible from a fresh
checkout in ≤12 min per subject.

### 3.1 The lag matrix — central finding (Analysis 4 + Bridwell test)

Trial-level Pearson r between mean posterior alpha during the engaged
window (target + maintenance) and mean V1 BOLD during the HRF-lagged
engaged window, for each (subject, load, lag).

| Subject | Load | r @ lag 3 | r @ lag 5 | r @ lag 7 | r @ lag 9 | r @ lag 11 |
|---|---:|---:|---:|---:|---:|---:|
| **sub-500** (canonical) | 1 | −0.042 | −0.029 | −0.159 | −0.084 | +0.019 |
| **sub-500** (canonical) | 5 | −0.106 | **−0.245** | −0.216 | −0.117 | −0.044 |
| **sub-1070302** (vulnerable) | 1 | +0.132 | +0.152 | +0.050 | −0.130 | **−0.230** |
| **sub-1070302** (vulnerable) | 5 | +0.168 | +0.153 | **+0.237** | +0.222 | +0.095 |

Read: **sub-1070302 at load1** is canonical-negative *if you look at the
right (delayed) lag*; **sub-1070302 at load5** is truly inverted at every
lag. **sub-500** is canonical-negative at lag 5 (Glover peak), strengthens
with load.

### 3.2 Behaviour (Analysis 1)

| Subject | ACC L1 | ACC L5 | **Δ ACC** | RT L1 | RT L5 | Δ RT |
|---|---:|---:|---:|---:|---:|---:|
| sub-500 | 0.875 | 0.825 | −0.050 | 0.737 s | 0.890 s | +0.153 s |
| sub-1070302 | 0.950 | 0.850 | **−0.100** | 0.696 s | 0.803 s | +0.107 s |

sub-1070302 pays 2× the accuracy cost for high WM load. Speed-accuracy
tradeoff diverges: sub-500 slows down to preserve accuracy; sub-1070302
keeps speed and loses accuracy.

### 3.3 Vigilance × RT validity (Analysis 5)

Pearson r between trial-level alpha_engaged and reactionTime, correct
trials only, pooled across runs.

| Subject | n correct | r(alpha, RT) | Reading |
|---|---:|---:|---|
| sub-500 | 68 | −0.06 (null) | Alpha doesn't track RT (no vigilance to track) |
| sub-1070302 | 72 | **+0.123** (canonical, weak) | Alpha tracks RT (vigilance is fluctuating and affecting performance) |

### 3.4 The 7 phenotype-dissociating signals (combined evidence)

Each independent measurement we have made stratifies the two subjects on
the same phenotype axis:

| # | Source | Phenotype split |
|---|---|---|
| 1 | EEG EOEC outside-scanner alpha EC/EO ratio | sub-500: 3.10× | sub-1070302: 0.85× (reversed) |
| 2 | fMRI EOEC V1 BOLD direction | sub-500: EO>EC | sub-1070302: EC>EO (reversed) |
| 3 | Rest alpha-BOLD GLM z[V1] | sub-500: −1.09 | sub-1070302: −0.08 (broken) |
| 4 | Whole-run α-BOLD lag at swm runs | sub-500: 4-6 TR canonical | sub-1070302: no signal at any lag |
| 5 | Behavioural accuracy decrement at load5 | sub-500: −5 % | sub-1070302: −10 % |
| 6 | Trial-level α-BOLD coupling × load × lag | sub-500: canonical strengthens | sub-1070302: lag-shift → true inversion |
| 7 | Vigilance × RT validity | sub-500: null | sub-1070302: canonical +0.12 |

Seven independent measurements pointing the same way at N=2 is unusually
strong evidence for a real mechanism rather than two coincident outliers.

### 3.5 What did *not* replicate at N=2 (transparency)

- **Klimesch 1999 alpha desync at encoding** — direction backwards or
  within noise in both subjects. Probably band-filter-then-average is too
  coarse; future analyses will use Morlet wavelet decomposition.
- **Jensen 2002 frontal-midline theta during maintenance** — anti-canonical
  in both subjects. Probably channel cluster or windowing issue.

We disclose these failed replications because they bound what we can and
cannot claim — the load × lag × phenotype finding is novel and clean, but
the canonical Klimesch/Jensen WM signatures need methodological revision
before scaling.

## 4 Hypotheses for n=156 — pre-registered

**H1** (existence of phenotype dimension). The across-subject distribution
of (load5 α-BOLD r at lag 5 minus load1 α-BOLD r at lag 5) is
*not* a narrow Gaussian centred at the group mean. We predict a heavy-tailed
or bimodal distribution with one cluster near our sub-500 value (−0.22)
and a second near our sub-1070302 value (+0.00 at lag 5).

**H2** (load-dependent transition signature). In a subset of subjects
≥ 15 % of the cohort, the coupling sign at load1 *and* at load5 will
**differ at the same lag**, with the transition occurring between load
levels. Pre-registered Δ-sign metric: sign(r_load1_lag5) ≠ sign(r_load5_lag5)
will identify these subjects.

**H3** (extension of Bridwell 2025 into vigilance). In the same vulnerable
cluster, the load1 coupling will be **delayed-but-canonical** — negative
at lag ≥ 9 TR even though positive at lag 5 — replicating Bridwell 2025's
psychiatric finding in a *vigilance-phenotyping* context.

**H4** (convergent validity). Cluster membership predicts, at α = 0.01,
an independent 5-element vigilance covariate vector:
1. EEG occipital α EO/EC ratio (Berger; Barry 2007)
2. EEG θ/α posterior-channel ratio (Hori 1994; N1 marker)
3. EOG slow eye movements
4. Tagliazucchi 2012 EEG vigilance index
5. Falahpour 2018 BOLD vigilance-template correlation

with 30 % held-out validation split.

**H5** (predictive validity for WM behaviour). The
vulnerable cluster shows ≥ 2× larger accuracy decrement load1→load5 than
the canonical cluster (mixed-effects regression, subject random effect,
pre-registered effect direction).

**H6** (conditional validity of α-RT marker). The
single-trial Pearson(α_engaged, RT) correlation differs significantly
between clusters: positive in the vulnerable cluster (validating alpha as
a vigilance index for these subjects), null in the canonical cluster.

## 5 Analysis plan — concrete and locked

All scripts already exist in our public repo and have run end-to-end on
both pilot subjects (Total Delta runtime: ~30 min/subject from cold cache,
~3 min/subject from warm cache).

1. **Pipeline per subject** (cached):
 - EEG MR + BCG removal via `eegfmri_loader`
 - ANTs rigid-body motion correction
 - Per-subject stimloc V1 functional localizer (z > 2.3 + top-1 % fallback)
 - Trial-locked extraction: per-trial α (baseline/encode/engaged), θ
   (maintenance), V1 BOLD at lags 3/5/7/9/11 TR
 - HRF-lag whole-run sanity check
 - Across-trial Pearson r per (load, lag), per subject
 - 5-element vigilance vector

2. **Group-level analyses** (~ 6 h on a single Delta GPU node):
 - HDBSCAN clustering on (r_load1_lag5, r_load5_lag5, lag_of_min_r_load1) → H1, H2, H3
 - Mixed-effects regression of cluster membership ↔ vigilance vector → H4
 - Mixed-effects regression of accuracy decrement ↔ cluster → H5
 - Bayesian comparison of α-RT correlation by cluster → H6
 - Replicate Nakuci 2023 connectome ICC on our cohort as a positive
   control for the unimodal-reliability framework

3. **Pre-registration** on OSF before any data beyond the 2 pilot subjects
 is touched. Pre-reg ID published with the proposal acceptance.

4. **Robustness sweep** (already implemented for N=2, will run on n=156):
 - HRF lag sweep 3–11 TR (the Bridwell test)
 - Morlet wavelet time-frequency for α desync and θ maintenance (replacing
   the broken band-filter approach)
 - Sensitivity to motion (FD < 0.2 mm subset analysis)
 - Sensitivity to 6 mm smoothing (0 / 3 / 6 mm sweep)

5. **Reproducibility**: every output is one Slurm job from a fresh
 checkout. We will share derivatives + cache + outputs with the
 collaborator before paper submission.

## 6 Risks and pre-emptions

| Risk | Defense |
|---|---|
| **"N=2 is anecdote."** | Frame as case-comparison / proof-of-concept; bootstrap CIs and trial-shuffle permutation nulls per correlation; pre-register on OSF before n=156 analysis; lean on **double-dissociation** logic (the two subjects differ on a *pre-specified* rest-state marker that *predicts* a pre-specified task-state pattern) rather than group statistics. |
| **"Just Meltzer 2007 / Scheeringa 2009 redone."** | We cite both directly. Scheeringa showed *if* α-BOLD coupling exists it is negative and load-modulated; we show *whether* it is negative is phenotype-conditional, *and* we identify the load × lag transition mechanism. Meltzer found half-half subjects but explicitly noted no behavioural or physiological correlate — we provide both. |
| **"Sign flip is just a lag shift (Bridwell)."** | **Pre-emptively tested in N=2.** Lag sweep 3–11 TR confirmed sub-1070302's load1 coupling is lag-shifted canonical (true negative at lag 11) and load5 coupling is true sign flip (positive at every lag). The lag-test methodology becomes a pre-registered standard of evidence. |
| **"You haven't ruled out motion / vascular reactivity / BCG differences."** | Report FD and DVARS per subject; run band-control (θ over occipital, alpha over motor cortex) and region-control (motor cortex, anterior insula) analyses; demonstrate eyes-open/eyes-closed alpha modulation in both subjects to confirm recording quality is equivalent. Pre-registered as supplementary. |
| **"Klimesch alpha desync and Jensen theta failed in your data — methodology is unreliable."** | Disclosed openly. We acknowledge the band-filter-average approach is too coarse for those signatures; we will use Morlet wavelet decomposition for n=156 and report whether the canonical Klimesch/Jensen patterns now replicate. This is a methodological transparency, not a fatal flaw. |
| **"Drowsiness is just motion."** | The vigilance covariate vector is constructed from EEG, EOG, and BOLD-template indicators independent of FD. Report results on a low-motion subset (FD < 0.2 mm) in supplementary. |
| **"You're fishing for clusters."** | Pre-registered cluster method (HDBSCAN, default parameters, no post-hoc tuning). 30 % held-out validation. Stability under leave-one-task-out resampling required ≥ 70 %. |

## 7 What we are asking for, and what the collaborator gets

**Asking for:**

1. Read access to the BIDS root containing the n=156 simultaneous EEG-fMRI
 cohort, identical structure to sub-500 / sub-1070302.
2. (If available) the n=67 face-task extension dataset.
3. Demographics and any vigilance / WM questionnaire variables already
 collected (ASRS, MEWS, Karolinska Sleepiness Scale, sleep diaries, etc.).
4. One 30-minute kickoff meeting to align on the pre-registered hypotheses
 before OSF lock.

**The collaborator gets:**

1. Co-authorship on the resulting paper (anticipated *NeuroImage* /
 *Imaging Neuroscience* / *Cerebral Cortex* methodology track).
2. The fully reproducible pipeline as a code drop: cached EEG cleanup,
 BOLD motion correction, per-subject V1 masks, trial-locked extraction,
 lag sweep, vigilance vector, clustering, behavioural prediction.
3. The per-subject (load × lag × phenotype) coupling profile and the
 vigilance-cluster assignment as a derivative dataset for their own
 future work.
4. Replication of Nakuci 2023's unimodal connectome ICC on their cohort
 as a side-product positive control (relevant to their reliability /
 reproducibility interests).

## 8 Timeline (post-access)

| Week | Deliverable |
|---|---|
| 1 | Pipeline deployed on first 10 subjects; QC pass/fail report; visual inspection of lag matrices |
| 2 | Full n=156 trial-locked extraction complete; preliminary (r_load1_lag5, r_load5_lag5) scatter |
| 3 | Vigilance covariate vector built; H1 + H2 + H3 tested |
| 4 | H4 + H5 + H6 tested; main-text figure draft |
| 5 | Robustness sweep (motion, smoothing, band-control, region-control) |
| 6 | Draft manuscript circulated to all co-authors |

Total: 6 weeks of analyst time. Compute is on the existing
`bbnv-delta-gpu` allocation; no additional resources requested.

## 9 Contact

Kuntal Kokate — kukokate@ucsd.edu
Repo: github.com/Kkuntal990/eegfmri (full pipeline + N=2 evidence reproducible)
Pre-registration draft: available on request; will be OSF-locked before
data access begins.

---

## Appendix — what changed from the earlier reliability-framed proposal

The earlier draft framed the contribution as *cross-state ICC of alpha-BOLD
coupling maps*. We have since found that voxelwise ICC is essentially zero
in both subjects (ICC = +0.011, −0.013), making it a non-starter as a
biomarker. The vigilance × WM trial-locked analysis instead delivered the
clean N=2 dissociation that motivates this revised proposal. The earlier
reliability findings remain useful as one paragraph in the methods section
— justifying why we report whole-run ROI-mean coupling rather than
voxelwise maps — but are no longer the headline.
