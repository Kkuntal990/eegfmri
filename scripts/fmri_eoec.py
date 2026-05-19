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

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
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


def classify(X: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str,
             pipeline) -> dict:
    """5-fold GroupKFold CV; groups = block index."""
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return {"name": name, "error": "not enough blocks for CV",
                "n_groups": int(n_groups)}
    gkf = GroupKFold(n_splits=n_splits)
    accs, baccs, cms = [], [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        pipeline.fit(X[tr], y[tr])
        pred = pipeline.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))
        cms.append(confusion_matrix(y[te], pred, labels=[0, 1]).tolist())
    return {
        "name": name,
        "n_splits": n_splits,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs)),
        "fold_accuracies": [float(a) for a in accs],
        "fold_confusion_matrices": cms,
    }


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

    log.info("Computing brain mask")
    mask_img = compute_epi_mask(bold_img)
    n_mask_voxels = int(mask_img.get_fdata().astype(bool).sum())
    log.info("  %d voxels in mask", n_mask_voxels)

    log.info("Masking + cleaning (detrend, z-score, bandpass 0.01-0.10 Hz)")
    masker = NiftiMasker(
        mask_img=mask_img,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.10,
        high_pass=0.01,
        t_r=TR,
    )
    ts = masker.fit_transform(bold_img)  # (T, n_voxels)
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
