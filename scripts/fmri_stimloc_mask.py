"""Per-subject functional visual-cortex localizer from the stimloc task.

The stimloc task shows the subject Sternberg-style dot stimuli (`loc_targ_l1`
and `loc_targ_l5`) interleaved with blank-screen baselines. Voxels whose BOLD
rises during stimulus events are by construction visual-responsive.

Pipeline per subject:

  1. Load BOLD .nii.gz with nibabel.
  2. Motion correction with ANTs (same routine as scripts/fmri_eoec.py).
  3. Build events DataFrame: every loc_targ_* trial is `stim_on`.
  4. nilearn FirstLevelModel: HRF-convolved GLM, cosine drift, 6 mm smoothing,
     FD + FD' as nuisance regressors.
  5. Contrast `stim_on` -> z-map.
  6. Threshold z > 3.1 (p < 0.001 one-tailed, uncorrected) -> binary mask.
  7. Save mask + z-map under
     results/stimloc_mask/sub-XXX/sub-XXX_ses-02_stimloc-visual-mask.nii.gz

The resulting masks are picked up automatically by scripts/fmri_eoec.py via
--stimloc-mask-dir, replacing the crude bottom-axial-quartile heuristic.

Usage:
    python scripts/fmri_stimloc_mask.py \
        --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/stimloc_mask
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import new_img_like
from nilearn.masking import compute_epi_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cache import get_motion_corrected_bold  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TR = 1.0
Z_THRESH = 2.3   # ~ p < 0.01 one-tailed, uncorrected
MIN_MASK_VOXELS = 100  # if fewer than this pass threshold, fall back to top 1%


def discover_subjects(bids_root: Path) -> list[str]:
    subs = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if (sub_dir / "ses-02" / "func" /
                f"{sub_dir.name}_ses-02_task-stimloc.nii.gz").exists():
            subs.append(sub_dir.name.replace("sub-", ""))
    return subs


def motion_correct(bold_path: Path, sub: str, n_trs_expected: int):
    """Disk-cached ANTs motion correction. Returns (nibabel img, FD)."""
    bold_img, fd = get_motion_corrected_bold(
        bold_path, sub, ses="02", task="stimloc",
    )
    if len(fd) < n_trs_expected:
        fd = np.concatenate([fd, np.zeros(n_trs_expected - len(fd))])
    return bold_img, fd


def build_events_df(events_tsv: Path) -> pd.DataFrame:
    """Collapse all loc_targ_* trials into a single `stim_on` regressor.

    The two subjects' events.tsv files have different schemas:
      - sub-500: 12 cols, durations in milliseconds (~2000)
      - sub-1070302: 8 cols, durations in seconds (~2.0)
    Auto-detect by median magnitude.
    """
    df = pd.read_csv(events_tsv, sep="\t")
    mask = df["value"].astype(str).str.startswith("loc_targ_")
    events = df[mask].copy()
    raw_dur = pd.to_numeric(events["duration"], errors="coerce")
    median_dur = float(raw_dur.median()) if len(raw_dur) else 0.0
    if median_dur > 50:  # almost certainly milliseconds
        dur_s = raw_dur / 1000.0
        log.info("  duration column detected as MILLISECONDS (median=%.1f)", median_dur)
    else:
        dur_s = raw_dur
        log.info("  duration column detected as SECONDS (median=%.2f)", median_dur)
    out = pd.DataFrame({
        # Use onset relative to MRI start (column already in seconds).
        "onset": pd.to_numeric(events["onsetRelToMRIstart"], errors="coerce"),
        "duration": dur_s,
        "trial_type": "stim_on",
    }).dropna()
    return out.reset_index(drop=True)


def localize_subject(sub: str, bids_root: Path, out_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("Subject sub-%s", sub)
    bold_path = bids_root / f"sub-{sub}" / "ses-02" / "func" / \
                f"sub-{sub}_ses-02_task-stimloc.nii.gz"
    events_path = bids_root / f"sub-{sub}" / "ses-02" / "eeg" / \
                  f"sub-{sub}_ses-02_task-stimloc_events.tsv"

    if not bold_path.exists() or not events_path.exists():
        log.warning("Missing data for sub-%s", sub)
        return {"subject": sub, "error": "missing BOLD or events"}

    bold_img = nib.load(str(bold_path))
    n_trs = bold_img.shape[3]
    log.info("  BOLD shape=%s", bold_img.shape)

    bold_img, fd = motion_correct(bold_path, sub, n_trs)

    events = build_events_df(events_path)
    log.info("  %d stim_on trials, mean duration %.2fs",
             len(events), events["duration"].mean())

    fd_d = np.concatenate([[0.0], np.diff(fd)])
    confounds = pd.DataFrame({"fd": fd, "fd_d": fd_d})

    mask_img = compute_epi_mask(bold_img)
    n_brain = int(mask_img.get_fdata().astype(bool).sum())
    log.info("  brain mask voxels: %d", n_brain)

    log.info("Fitting FirstLevelModel (HRF=glover, drift=cosine, smooth=6mm)")
    glm = FirstLevelModel(
        t_r=TR,
        hrf_model="glover",
        drift_model="cosine",
        high_pass=0.01,
        mask_img=mask_img,
        smoothing_fwhm=6.0,
        standardize=False,
        minimize_memory=False,
        verbose=0,
    )
    glm.fit(bold_img, events=events, confounds=confounds)
    z_map = glm.compute_contrast("stim_on", output_type="z_score")
    z_data = z_map.get_fdata()
    log.info("  z-map range: [%.2f, %.2f]", float(z_data.min()), float(z_data.max()))

    bin_mask = (z_data > Z_THRESH).astype(np.uint8)
    # Intersect with brain mask so we never include out-of-brain voxels.
    bin_mask = bin_mask * mask_img.get_fdata().astype(np.uint8)
    n_voxels = int(bin_mask.sum())
    log.info("  visual mask voxels (z > %.1f): %d (%.1f%% of brain)",
             Z_THRESH, n_voxels, 100.0 * n_voxels / max(n_brain, 1))

    if n_voxels < MIN_MASK_VOXELS:
        log.warning("  only %d voxels at z>%.1f (<%d) -- falling back to top-1%% z",
                    n_voxels, Z_THRESH, MIN_MASK_VOXELS)
        top_n = max(int(0.01 * n_brain), 100)
        flat = z_data * mask_img.get_fdata().astype(np.uint8)
        cutoff = np.partition(flat.ravel(), -top_n)[-top_n]
        bin_mask = (flat >= cutoff).astype(np.uint8)
        n_voxels = int(bin_mask.sum())
        log.info("  fallback mask voxels: %d (cutoff z=%.2f)",
                 n_voxels, float(cutoff))

    sub_out = out_dir / f"sub-{sub}"
    sub_out.mkdir(parents=True, exist_ok=True)
    mask_path = sub_out / f"sub-{sub}_ses-02_stimloc-visual-mask.nii.gz"
    zmap_path = sub_out / f"sub-{sub}_ses-02_stimloc-zmap.nii.gz"
    nib.save(new_img_like(bold_img, bin_mask), str(mask_path))
    nib.save(z_map, str(zmap_path))
    log.info("  saved %s", mask_path)

    return {
        "subject": sub,
        "n_trs": int(n_trs),
        "n_events": int(len(events)),
        "z_threshold": Z_THRESH,
        "n_brain_voxels": n_brain,
        "n_visual_voxels": n_voxels,
        "z_max": float(z_data.max()),
        "fd_mean_mm": float(fd.mean()),
        "fd_max_mm": float(fd.max()),
        "mask_path": str(mask_path),
        "zmap_path": str(zmap_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data",
                    type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subs = args.subjects or discover_subjects(args.bids_root)
    log.info("Subjects: %s", subs)

    results = [localize_subject(s, args.bids_root, args.output_dir) for s in subs]
    out_path = args.output_dir / "stimloc_mask_results.json"
    out_path.write_text(json.dumps({"per_subject": results}, indent=2))
    log.info("Wrote %s", out_path)

    print("\n" + "=" * 70)
    print(f"{'Subject':<12} {'Events':>8} {'BrainVox':>10} {'VisualVox':>12} {'%brain':>8}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['subject']:<12} ERROR: {r['error']}")
            continue
        pct = 100.0 * r["n_visual_voxels"] / max(r["n_brain_voxels"], 1)
        print(f"{r['subject']:<12} {r['n_events']:>8} {r['n_brain_voxels']:>10} "
              f"{r['n_visual_voxels']:>12} {pct:>7.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
