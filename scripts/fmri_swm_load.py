"""Sternberg working-memory load classification (load1 vs load5).

Within-subject, leave-one-run-out CV across swm_run-1 and swm_run-2. For each
trial:

  EEG features:
    - encode theta: frontal-midline (E11/E12) Hilbert log-power, 4-7 Hz,
      averaged over the 2 s target window
    - maintain alpha: posterior (E62/E67/E70/E72/E75/E76/E83/E84) Hilbert
      log-power, 8-12 Hz, averaged over the variable 1.5-10 s maintenance
      window

  fMRI features:
    - whole-brain BOLD vector: mean over the maintenance window with
      +5 TR HRF lag, masked by the EPI brain mask -> PCA(50) for the
      whole-brain classifier
    - V1 BOLD: same window, restricted to the per-subject stimloc-derived
      visual mask, averaged across voxels

Classifiers (LinearSVC inside a paired-block-style cross-run CV):
  - EEG only       (2 features: theta + alpha)
  - fMRI V1 mean   (1 feature)
  - fMRI whole-brain  (PCA50 of voxel means)
  - Joint EEG + fMRI  (concatenated)

We restrict to accurate trials (accuracy == 1) to keep the WM signal clean;
output also reports the trial counts so we can sanity-check.

Usage:
    python scripts/fmri_swm_load.py \
        --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/fmri_swm_load
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cache import get_cleaned_eeg_for_task, get_motion_corrected_bold  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TR = 1.0
HRF_LAG_TR = 5
ENCODE_DUR_S = 2.0
RUNS = ["1", "2"]

FRONTAL_CHS = ["E11", "E12", "E5", "E6"]   # frontal-midline theta cluster
POSTERIOR_CHS = ["E62", "E67", "E70", "E72", "E75", "E76", "E83", "E84"]


# ---------------------------------------------------------------------------
# Trial parsing from events.tsv
# ---------------------------------------------------------------------------

def parse_swm_trials(events_tsv: Path, run_id: str) -> list[dict]:
    """Return list of accurate trials with target onset, load, maint duration."""
    df = pd.read_csv(events_tsv, sep="\t")
    val = df["value"].astype(str)
    targ_mask = val.str.startswith("SDRT_targ_")
    targs = df[targ_mask]
    trials = []
    for _, row in targs.iterrows():
        try:
            acc = int(row["accuracy"])
            if acc != 1:
                continue
            load = 0 if str(row["loadName"]) == "load1" else 1   # binary
            onset = float(row["onsetRelToMRIstart"])
            maint = float(row["maintDuration"])
        except (ValueError, KeyError, TypeError):
            continue
        trials.append({
            "run": run_id,
            "onset_s": onset,
            "load": load,
            "maint_s": maint,
        })
    return trials


# ---------------------------------------------------------------------------
# EEG feature extraction
# ---------------------------------------------------------------------------

def _safe_picks(raw, candidates):
    return [ch for ch in candidates if ch in raw.ch_names]


def extract_eeg_features(raw, trials: list[dict]) -> np.ndarray:
    """Per trial: [encode_theta_log_power, maintain_alpha_log_power]."""
    sfreq = raw.info["sfreq"]
    frontal = _safe_picks(raw, FRONTAL_CHS)
    posterior = _safe_picks(raw, POSTERIOR_CHS)
    if not frontal or not posterior:
        raise ValueError(
            f"Missing channels — frontal:{frontal}, posterior:{posterior}"
        )
    log.info("    EEG: frontal=%s posterior=%s", frontal, posterior)

    raw_theta = raw.copy().pick(frontal).filter(4, 7, verbose=False)
    raw_alpha = raw.copy().pick(posterior).filter(8, 12, verbose=False)
    theta_env = np.abs(hilbert(raw_theta.get_data(), axis=1))   # (Cθ, T)
    alpha_env = np.abs(hilbert(raw_alpha.get_data(), axis=1))   # (Cα, T)

    n_samples = theta_env.shape[1]
    feats = []
    for tr in trials:
        i0 = int(round(tr["onset_s"] * sfreq))
        i1 = int(round((tr["onset_s"] + ENCODE_DUR_S) * sfreq))
        m0 = i1
        m1 = int(round((tr["onset_s"] + ENCODE_DUR_S + tr["maint_s"]) * sfreq))
        if i0 < 0 or m1 > n_samples or i1 <= i0 or m1 <= m0:
            feats.append([np.nan, np.nan])
            continue
        theta = np.log10(theta_env[:, i0:i1].mean() + 1e-12)
        alpha = np.log10(alpha_env[:, m0:m1].mean() + 1e-12)
        feats.append([theta, alpha])
    return np.asarray(feats, dtype=float)


# ---------------------------------------------------------------------------
# fMRI feature extraction
# ---------------------------------------------------------------------------

def extract_fmri_features(bold_img, fd: np.ndarray, trials: list[dict],
                          v1_mask_img, brain_mask_img=None):
    """Return (X_brain, X_v1, mask_voxel_count, brain_mask_img).

    X_brain: (n_trials, n_brain_voxels) — mean BOLD over each trial's
              maintenance window (+ HRF lag), cleaned with motion confounds.
    X_v1:    (n_trials, 1) — mean BOLD over the stimloc V1 mask in the
              same window.

    If `brain_mask_img` is supplied, it's reused (this guarantees the same
    voxel space across runs of the same subject); otherwise it's computed
    from this BOLD.
    """
    n_trs = bold_img.shape[3]
    if brain_mask_img is None:
        mask_img = compute_epi_mask(bold_img)
    else:
        mask_img = brain_mask_img

    fd_d = np.concatenate([[0.0], np.diff(fd)])
    confounds = pd.DataFrame({"fd": fd, "fd_d": fd_d})

    masker = NiftiMasker(
        mask_img=mask_img,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.10,
        high_pass=0.01,
        t_r=TR,
    )
    ts = masker.fit_transform(bold_img, confounds=confounds)   # (n_trs, V)
    log.info("    BOLD ts shape=%s", ts.shape)

    # Resample V1 mask to BOLD grid; find indices in the masker output.
    v1_resampled = resample_to_img(
        v1_mask_img, bold_img, interpolation="nearest",
        force_resample=True, copy_header=True,
    )
    v1_bool = v1_resampled.get_fdata().astype(bool) & mask_img.get_fdata().astype(bool)
    brain_bool = mask_img.get_fdata().astype(bool)
    v1_in_brain = v1_bool[brain_bool]   # bool over masker output dimension
    n_v1 = int(v1_in_brain.sum())
    log.info("    V1 voxels in brain mask: %d", n_v1)

    X_brain, X_v1 = [], []
    for tr in trials:
        m_start_tr = tr["onset_s"] + ENCODE_DUR_S
        m_end_tr = tr["onset_s"] + ENCODE_DUR_S + tr["maint_s"]
        i = int(round(m_start_tr / TR)) + HRF_LAG_TR
        j = int(round(m_end_tr / TR)) + HRF_LAG_TR
        i = max(i, 0); j = min(j, n_trs)
        if j <= i:
            X_brain.append(np.full(ts.shape[1], np.nan))
            X_v1.append(np.array([np.nan]))
            continue
        window = ts[i:j].mean(axis=0)
        X_brain.append(window)
        if n_v1 > 0:
            X_v1.append(np.array([window[v1_in_brain].mean()]))
        else:
            X_v1.append(np.array([np.nan]))
    return np.asarray(X_brain), np.asarray(X_v1), n_v1, mask_img


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

def cross_run_eval(X_run1, X_run2, y_run1, y_run2, name, pipeline) -> dict:
    """Train-on-run-1 / test-on-run-2 and vice versa; report mean accuracy."""
    accs, baccs, cms, sizes = [], [], [], []
    for train_X, train_y, test_X, test_y in [
        (X_run1, y_run1, X_run2, y_run2),
        (X_run2, y_run2, X_run1, y_run1),
    ]:
        pipeline.fit(train_X, train_y)
        pred = pipeline.predict(test_X)
        accs.append(accuracy_score(test_y, pred))
        baccs.append(balanced_accuracy_score(test_y, pred))
        cms.append(confusion_matrix(test_y, pred, labels=[0, 1]).tolist())
        sizes.append({"n_train": len(train_y), "n_test": len(test_y)})
    return {
        "name": name,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs)),
        "fold_accuracies": [float(a) for a in accs],
        "fold_confusion_matrices": cms,
        "fold_sizes": sizes,
    }


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------

def run_subject(sub: str, bids_root: Path, out_dir: Path,
                stimloc_mask_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("Subject sub-%s", sub)

    # Load V1 mask once
    v1_mask_path = stimloc_mask_dir / f"sub-{sub}" / \
                   f"sub-{sub}_ses-02_stimloc-visual-mask.nii.gz"
    if not v1_mask_path.exists():
        return {"subject": sub, "error": "missing stimloc V1 mask"}
    v1_mask_img = nib.load(str(v1_mask_path))
    log.info("V1 mask path: %s", v1_mask_path.name)

    # Load EEG once across both runs (cached as .fif after first run).
    log.info("Loading cleaned EEG (sub-%s, swm)", sub)
    raws_by_run_raw = get_cleaned_eeg_for_task(
        bids_root, sub, ses="02", task="swm",
    )
    # Normalise keys to strings (the cache may return None for single-run tasks).
    raws_by_run = {str(k): v for k, v in raws_by_run_raw.items() if k is not None}
    if set(raws_by_run.keys()) != set(RUNS):
        return {"subject": sub,
                "error": f"missing EEG runs (have: {list(raws_by_run.keys())})"}

    # Per-run feature extraction. Use a shared brain mask across runs so
    # cross-run classifier dimensions match (compute_epi_mask is data-driven
    # and gives slightly different masks per run otherwise).
    per_run = {}
    shared_brain_mask = None
    for run in RUNS:
        log.info("--- run %s ---", run)
        bold_path = bids_root / f"sub-{sub}" / "ses-02" / "func" / \
                    f"sub-{sub}_ses-02_task-swm_run-{run}.nii.gz"
        events_tsv = bids_root / f"sub-{sub}" / "ses-02" / "eeg" / \
                     f"sub-{sub}_ses-02_task-swm_run-{run}_events.tsv"
        if not bold_path.exists() or not events_tsv.exists():
            return {"subject": sub, "error": f"missing run-{run} data"}

        trials = parse_swm_trials(events_tsv, run)
        n1 = sum(1 for t in trials if t["load"] == 0)
        n5 = sum(1 for t in trials if t["load"] == 1)
        log.info("  %d accurate trials (load1=%d, load5=%d)",
                 len(trials), n1, n5)
        if min(n1, n5) < 3:
            return {"subject": sub, "error": f"too few trials in run-{run}"}

        bold_img, fd = get_motion_corrected_bold(
            bold_path, sub, ses="02", task="swm", run=run,
        )
        n_trs = bold_img.shape[3]
        log.info("  BOLD shape=%s, FD mean=%.3f mm", bold_img.shape,
                 float(fd.mean()))

        X_eeg = extract_eeg_features(raws_by_run[run], trials)
        X_brain, X_v1, n_v1, mask_used = extract_fmri_features(
            bold_img, fd, trials, v1_mask_img,
            brain_mask_img=shared_brain_mask,
        )
        if shared_brain_mask is None:
            shared_brain_mask = mask_used
            log.info("    using this mask for both runs (n voxels=%d)",
                     int(mask_used.get_fdata().astype(bool).sum()))
        y = np.asarray([t["load"] for t in trials])

        # Drop trials with any NaN feature
        keep = ~np.any(np.isnan(X_eeg), axis=1) & \
               ~np.any(np.isnan(X_brain), axis=1) & \
               ~np.any(np.isnan(X_v1), axis=1)
        dropped = int((~keep).sum())
        log.info("  dropped %d trials with NaN features (out of %d)",
                 dropped, len(trials))
        per_run[run] = {
            "X_eeg": X_eeg[keep],
            "X_brain": X_brain[keep],
            "X_v1": X_v1[keep],
            "y": y[keep],
            "n_v1": n_v1,
            "fd_mean_mm": float(fd.mean()),
            "n_trials_kept": int(keep.sum()),
            "n_load1": int((y[keep] == 0).sum()),
            "n_load5": int((y[keep] == 1).sum()),
        }

    # ---- Classifiers via leave-one-run-out CV ----
    Xe1, Xe2 = per_run["1"]["X_eeg"], per_run["2"]["X_eeg"]
    Xb1, Xb2 = per_run["1"]["X_brain"], per_run["2"]["X_brain"]
    Xv1, Xv2 = per_run["1"]["X_v1"], per_run["2"]["X_v1"]
    y1, y2 = per_run["1"]["y"], per_run["2"]["y"]

    classifiers = []
    classifiers.append(cross_run_eval(
        Xe1, Xe2, y1, y2, "EEG_only(theta+alpha)",
        make_pipeline(StandardScaler(),
                      LinearSVC(C=1.0, class_weight="balanced",
                                max_iter=5000)),
    ))
    if per_run["1"]["n_v1"] > 0 and per_run["2"]["n_v1"] > 0:
        classifiers.append(cross_run_eval(
            Xv1, Xv2, y1, y2, "fMRI_V1_mean",
            make_pipeline(StandardScaler(),
                          LinearSVC(C=1.0, class_weight="balanced",
                                    max_iter=5000)),
        ))
    classifiers.append(cross_run_eval(
        Xb1, Xb2, y1, y2, "fMRI_wholebrain_PCA50",
        make_pipeline(StandardScaler(with_mean=False),
                      PCA(n_components=min(50, Xb1.shape[0] - 1),
                          random_state=0),
                      LinearSVC(C=1.0, class_weight="balanced",
                                max_iter=5000)),
    ))
    Xj1 = np.hstack([Xe1, Xv1, Xb1])
    Xj2 = np.hstack([Xe2, Xv2, Xb2])
    classifiers.append(cross_run_eval(
        Xj1, Xj2, y1, y2, "Joint_EEG+V1+wholebrain",
        make_pipeline(StandardScaler(with_mean=False),
                      PCA(n_components=min(50, Xj1.shape[0] - 1),
                          random_state=0),
                      LinearSVC(C=1.0, class_weight="balanced",
                                max_iter=5000)),
    ))

    return {
        "subject": sub,
        "runs": {
            run: {
                k: v for k, v in per_run[run].items()
                if k not in ("X_eeg", "X_brain", "X_v1", "y")
            } for run in RUNS
        },
        "classifiers": classifiers,
    }


# ---------------------------------------------------------------------------
# Subject discovery + main
# ---------------------------------------------------------------------------

def discover_subjects(bids_root: Path) -> list[str]:
    subs = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        ok = all(
            (sub_dir / "ses-02" / "func" /
             f"{sub_dir.name}_ses-02_task-swm_run-{r}.nii.gz").exists()
            for r in RUNS
        )
        if ok:
            subs.append(sub_dir.name.replace("sub-", ""))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data", type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/fmri_swm_load",
                    type=Path)
    ap.add_argument("--stimloc-mask-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subs = args.subjects or discover_subjects(args.bids_root)
    log.info("Subjects: %s", subs)

    results = [run_subject(s, args.bids_root, args.output_dir,
                           args.stimloc_mask_dir) for s in subs]
    (args.output_dir / "swm_load_results.json").write_text(
        json.dumps({"per_subject": results}, indent=2)
    )

    print("\n" + "=" * 70)
    print(f"{'Subject':<12} {'Classifier':<28} {'Acc':>8} {'BalAcc':>8}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['subject']:<12} ERROR: {r['error']}")
            continue
        for c in r["classifiers"]:
            if "error" in c:
                continue
            print(f"{r['subject']:<12} {c['name']:<28} "
                  f"{c['accuracy_mean']:>7.1%} {c['balanced_accuracy_mean']:>7.1%}")
    print("=" * 70)
    print("Chance = 50 %. Above-chance load1-vs-load5 decoding = WM "
          "load signal present.")


if __name__ == "__main__":
    main()
