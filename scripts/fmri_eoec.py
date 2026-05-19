"""Within-subject fMRI Eyes-Open vs Eyes-Closed classification.

Counterpart to scripts/validate_eeg.py. Pipeline:

  1. Load BOLD .nii.gz with nibabel.
  2. Compute a brain mask (compute_epi_mask) and extract voxel time-series
     with NiftiMasker -- detrend, standardize, band-pass [0.01, 0.10] Hz.
  3. Parse events.tsv for ECstart / EOstart markers (30 s blocks).
  4. Label every TR as EC=0 / EO=1, drop the first 5 TRs of each block to
     absorb the HRF rise, and drop transition TRs.
  5. Within-subject classification with GroupKFold (groups = block index, so
     adjacent autocorrelated TRs never leak across folds):
         (a) PCA(50) + LinearSVC
         (b) LinearSVC on the raw whole-brain voxel vector
         (c) Logistic regression on the mean BOLD of posterior voxels (rough
             visual-cortex proxy: lowest 25 % of axial slices in mask)
  6. Save per-subject results to JSON.

Usage:
    python scripts/fmri_eoec.py --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
                                --output-dir /projects/bbnv/kkokate/eegfmri/results/fmri_eoec
"""

import argparse
import json
import logging
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TR = 1.0           # seconds, from JSON sidecar
HRF_LAG_TRS = 5    # drop first 5 TRs of each block to absorb hemodynamic rise
BLOCK_DUR_S = 30.0


def discover_subjects(bids_root: Path) -> list[str]:
    """Return subject IDs that have a ses-02 task-eoec BOLD file."""
    subs = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if (sub_dir / "ses-02" / "func" /
                f"{sub_dir.name}_ses-02_task-eoec.nii.gz").exists():
            subs.append(sub_dir.name.replace("sub-", ""))
    return subs


def parse_blocks(events_tsv: Path) -> list[tuple[float, str]]:
    """Return [(onset_s, label), ...] for each 30 s block.

    onset_s is seconds from MRI scan start (column `onsetRelToMRIstart`),
    label is "EC" or "EO".
    """
    df = pd.read_csv(events_tsv, sep="\t")
    # Block markers
    mask = df["value"].isin(["ECstart", "EOstart"])
    blocks = []
    for _, row in df[mask].iterrows():
        onset = float(row["onsetRelToMRIstart"])
        label = "EC" if row["value"] == "ECstart" else "EO"
        blocks.append((onset, label))
    return blocks


def label_trs(n_trs: int, blocks: list[tuple[float, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Label each TR.

    Returns:
        y:       (n_trs,) array, 0=EC, 1=EO, -1=ignore
        groups:  (n_trs,) array, block index for the TR (-1 if ignored)
    """
    y = np.full(n_trs, -1, dtype=int)
    groups = np.full(n_trs, -1, dtype=int)
    for bidx, (onset_s, label) in enumerate(blocks):
        first_tr = int(np.ceil(onset_s / TR)) + HRF_LAG_TRS  # skip HRF rise
        last_tr  = int(np.floor((onset_s + BLOCK_DUR_S) / TR))
        if first_tr >= n_trs:
            continue
        last_tr = min(last_tr, n_trs - 1)
        y[first_tr:last_tr + 1] = 0 if label == "EC" else 1
        groups[first_tr:last_tr + 1] = bidx
    return y, groups


def posterior_voxel_indices(mask_img) -> np.ndarray:
    """Bottom 25 % of axial slices inside the brain mask.

    Heuristic for occipital cortex without an atlas / MNI registration.
    """
    mask = mask_img.get_fdata().astype(bool)
    z_in_mask = np.where(mask.any(axis=(0, 1)))[0]
    if len(z_in_mask) == 0:
        return np.array([], dtype=int)
    z_cutoff = z_in_mask[len(z_in_mask) // 4]  # bottom quartile
    posterior_mask = mask.copy()
    posterior_mask[:, :, z_cutoff:] = False
    # NiftiMasker flattens in Fortran (column-major) order; mirror that.
    flat_mask = posterior_mask.flatten(order="F")
    full_mask = mask.flatten(order="F")
    return np.where(flat_mask[full_mask])[0]


def paired_block_folds(groups: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """3-fold CV pairing consecutive blocks so each test fold has 1 EC + 1 EO.

    Assumes blocks alternate EC, EO, EC, EO, ... For 6 blocks we get 3 folds
    of (1 EC + 1 EO) test, with the remaining 4 blocks for training.
    """
    unique_blocks = sorted(np.unique(groups[groups >= 0]))
    folds = []
    for i in range(0, len(unique_blocks) - 1, 2):
        test_blocks = [unique_blocks[i], unique_blocks[i + 1]]
        test_mask = np.isin(groups, test_blocks)
        train_mask = (groups >= 0) & ~test_mask
        # Sanity: both classes must be present in the test fold
        if len(np.unique(y[test_mask])) < 2:
            continue
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds


def classify(X: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str,
             pipeline) -> dict:
    """Paired-block CV: each fold tests on (1 EC + 1 EO) block, trains on the rest."""
    folds = paired_block_folds(groups, y)
    if len(folds) < 2:
        return {"name": name, "error": "not enough mixed-class fold pairs",
                "n_folds": len(folds)}
    accs, baccs, cms = [], [], []
    for tr_idx, te_idx in folds:
        pipeline.fit(X[tr_idx], y[tr_idx])
        pred = pipeline.predict(X[te_idx])
        accs.append(accuracy_score(y[te_idx], pred))
        baccs.append(balanced_accuracy_score(y[te_idx], pred))
        cms.append(confusion_matrix(y[te_idx], pred, labels=[0, 1]).tolist())
    return {
        "name": name,
        "n_folds": len(folds),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs)),
        "fold_accuracies": [float(a) for a in accs],
        "fold_confusion_matrices": cms,
    }


def motion_correct(bold_img):
    """Rigid-body motion correction via ANTs.

    Returns:
        corrected_img:  nibabel Nifti1Image, same shape as input
        fd:             (T,) framewise displacement
        params:         (T, 6) per-volume rigid-body parameters
                        (3 rotations rad, 3 translations mm) or None if
                        antspyx didn't expose them
    """
    log.info("Motion correction (ANTs rigid)")
    ants_img = ants.from_nibabel(bold_img)
    mc = ants.motion_correction(ants_img)
    corrected_img = ants.to_nibabel(mc["motion_corrected"])

    fd = np.asarray(mc.get("FD", []), dtype=float)
    if fd.size == 0:
        fd = np.zeros(bold_img.shape[3], dtype=float)
    log.info("  FD: mean=%.3f mm, max=%.3f mm",
             float(fd.mean()), float(fd.max()))

    # motion_parameters in modern antspyx is a pandas DataFrame with the
    # 6 rigid params per volume (3 rotations + 3 translations).
    params = mc.get("motion_parameters", None)
    if isinstance(params, pd.DataFrame):
        params = params.values
    elif params is not None and hasattr(params, "__len__"):
        try:
            params = np.asarray(params, dtype=float)
        except Exception:
            params = None
    return corrected_img, fd, params


def run_subject(sub: str, bids_root: Path, out_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("Subject sub-%s", sub)
    bold_path = bids_root / f"sub-{sub}" / "ses-02" / "func" / \
                f"sub-{sub}_ses-02_task-eoec.nii.gz"
    events_path = bids_root / f"sub-{sub}" / "ses-02" / "eeg" / \
                  f"sub-{sub}_ses-02_task-eoec_events.tsv"

    if not bold_path.exists():
        log.warning("Missing %s", bold_path)
        return {"subject": sub, "error": "missing BOLD"}
    if not events_path.exists():
        log.warning("Missing %s", events_path)
        return {"subject": sub, "error": "missing events"}

    log.info("Loading %s", bold_path.name)
    bold_img = nib.load(str(bold_path))
    log.info("  shape=%s, zooms=%s", bold_img.shape, bold_img.header.get_zooms())
    n_trs = bold_img.shape[3]

    bold_img, fd, motion_params = motion_correct(bold_img)

    # Build a confound matrix: 6 motion params + FD + their first derivatives.
    if motion_params is not None and motion_params.shape[0] == n_trs:
        mp = motion_params
    else:
        log.warning("motion_parameters not available — using FD-only confounds")
        mp = fd.reshape(-1, 1)
    mp_d = np.vstack([np.zeros((1, mp.shape[1])), np.diff(mp, axis=0)])
    confounds = np.hstack([mp, mp_d, fd.reshape(-1, 1)])
    log.info("  confound matrix shape=%s", confounds.shape)

    log.info("Computing brain mask")
    mask_img = compute_epi_mask(bold_img)
    n_mask_voxels = int(mask_img.get_fdata().astype(bool).sum())
    log.info("  %d voxels in mask", n_mask_voxels)

    log.info("Masking + cleaning (detrend, z-score, bandpass 0.01-0.10 Hz, regress motion)")
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.10,
        high_pass=0.01,
        t_r=TR,
    )
    ts = masker.fit_transform(bold_img, confounds=confounds)  # (T, n_voxels)
    log.info("  ts shape=%s", ts.shape)

    log.info("Parsing block labels")
    blocks = parse_blocks(events_path)
    log.info("  %d blocks: %s", len(blocks),
             [(round(o, 1), l) for o, l in blocks])
    y_full, groups_full = label_trs(n_trs, blocks)
    keep = y_full >= 0
    X = ts[keep]
    y = y_full[keep]
    groups = groups_full[keep]
    n_ec = int((y == 0).sum())
    n_eo = int((y == 1).sum())
    log.info("  kept %d TRs (EC=%d, EO=%d, blocks=%d)",
             len(y), n_ec, n_eo, len(np.unique(groups)))

    posterior_idx = posterior_voxel_indices(mask_img)
    log.info("  posterior-quartile voxels: %d", len(posterior_idx))

    classifiers = []
    # (a) PCA + LinearSVC on full brain
    classifiers.append(classify(
        X, y, groups, "PCA50+LinearSVC",
        make_pipeline(StandardScaler(with_mean=False),
                      PCA(n_components=50, random_state=0),
                      LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)),
    ))
    # (b) LinearSVC directly on voxels
    classifiers.append(classify(
        X, y, groups, "LinearSVC_voxels",
        make_pipeline(StandardScaler(with_mean=False),
                      LinearSVC(C=0.01, class_weight="balanced", max_iter=5000)),
    ))
    # (c) Mean signal across posterior voxels (visual-cortex proxy)
    if len(posterior_idx) > 0:
        X_post_mean = ts[:, posterior_idx].mean(axis=1, keepdims=True)[keep]
        classifiers.append(classify(
            X_post_mean, y, groups, "PosteriorMean+LogReg",
            make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced",
                                             max_iter=5000)),
        ))

    return {
        "subject": sub,
        "n_trs_total": int(n_trs),
        "n_trs_kept": int(len(y)),
        "n_ec": n_ec,
        "n_eo": n_eo,
        "n_blocks": int(len(np.unique(groups))),
        "n_mask_voxels": n_mask_voxels,
        "n_posterior_voxels": int(len(posterior_idx)),
        "motion": {
            "fd_mean_mm": float(fd.mean()),
            "fd_max_mm": float(fd.max()),
            "fd_p95_mm": float(np.percentile(fd, 95)),
        },
        "classifiers": classifiers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data",
                    type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/fmri_eoec",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Restrict to these subject IDs (without 'sub-' prefix)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subs = args.subjects or discover_subjects(args.bids_root)
    log.info("Subjects: %s", subs)

    results = {"per_subject": []}
    for sub in subs:
        results["per_subject"].append(run_subject(sub, args.bids_root,
                                                  args.output_dir))

    out_path = args.output_dir / "fmri_eoec_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)

    # Console summary table
    print("\n" + "=" * 70)
    print(f"{'Subject':<12} {'Classifier':<25} {'Acc':>8} {'BalAcc':>8}")
    print("-" * 70)
    for r in results["per_subject"]:
        if "error" in r:
            print(f"{r['subject']:<12} ERROR: {r['error']}")
            continue
        for c in r["classifiers"]:
            if "error" in c:
                continue
            print(f"{r['subject']:<12} {c['name']:<25} "
                  f"{c['accuracy_mean']:>7.1%} {c['balanced_accuracy_mean']:>7.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
