"""Alpha-BOLD coupling GLM on the rest task (Goldman 2002 / Ingram 2024 style).

For each subject:
  1. Load the 6-min rest BOLD and motion-correct with ANTs.
  2. Load the cleaned EEG (rest) via the existing eegfmri_loader.
  3. Compute occipital alpha (8-12 Hz) Hilbert envelope, smooth 5 s,
     downsample to TR resolution, z-score.
  4. HRF-convolve the alpha envelope (Glover) -> parametric regressor.
  5. Build a per-voxel GLM with the alpha regressor + motion confounds +
     cosine drift, fit on the whole-brain BOLD.
  6. Compute the alpha contrast -> z-map.
  7. Report mean z inside the stimloc-derived V1 mask.

Hypothesis (Goldman 2002, replicated in Ingram 2024): canonical alpha-BOLD
coupling is *negative* in visual cortex (high alpha = low BOLD). For
sub-1070302 we expect attenuated / reversed coupling -- the same anomaly
we documented in EOEC.

Output: per-subject z-map in results/fmri_alpha_bold/sub-XXX/ + a JSON
summary.

Usage:
    python scripts/fmri_eeg_alpha_glm.py \
        --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/fmri_alpha_bold
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import (
    FirstLevelModel,
    compute_regressor,
    make_first_level_design_matrix,
)
from nilearn.image import new_img_like, resample_to_img
from nilearn.masking import compute_epi_mask
from scipy.signal import hilbert

# Allow `from eegfmri_loader import ...` when running from the repo root.
sys.path.insert(0, "/projects/bbnv/kkokate/eegfmri")
from eegfmri_loader import load_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TR = 1.0
ALPHA_BAND = (8.0, 12.0)
SMOOTH_WIN_S = 5.0
# Occipital + parietal channels in the 128-ch EGI HydroCel cap. Same
# channels used in validate_eeg.py for the Berger contrast.
OCCIPITAL_CHS = ["E62", "E67", "E70", "E72", "E75", "E76", "E83", "E84"]


def motion_correct(bold_path: Path, n_trs_expected: int):
    """Rigid-body motion correction via ANTs. Returns (nibabel img, FD)."""
    log.info("Motion correction (ANTs rigid)")
    ants_img = ants.image_read(str(bold_path))
    mc = ants.motion_correction(ants_img)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        tmp_path = f.name
    ants.image_write(mc["motion_corrected"], tmp_path)
    corrected_img = nib.load(tmp_path)
    fd = np.asarray(mc.get("FD", []), dtype=float)
    if fd.size == 0:
        fd = np.zeros(n_trs_expected, dtype=float)
    log.info("  FD: mean=%.3f mm, max=%.3f mm",
             float(fd.mean()), float(fd.max()))
    return corrected_img, fd


def load_rest_eeg(bids_root: Path, sub: str):
    """Load the cleaned (MR + BCG removed) ses-02 rest EEG via the BIDS loader."""
    log.info("Loading cleaned EEG (sub-%s, rest)", sub)
    dataset = load_dataset(
        bids_root,
        subjects=[sub],
        sessions=["02"],
        tasks=["rest"],
        preprocess=True,
        resample_freq=250.0,
        pick_eeg_only=True,
        mr_artifact_removal=True,
        bcg_artifact_removal=True,
        n_jobs=1,
    )
    if len(dataset.datasets) == 0:
        return None
    return dataset.datasets[0].raw


def occipital_alpha_envelope(raw, target_tr_s: float = TR) -> np.ndarray:
    """Compute occipital alpha envelope at TR resolution.

    Steps: pick channels -> 8-12 Hz band-pass -> Hilbert amplitude per channel
    -> square (power) -> log10 -> mean across channels -> 5 s box-car smooth
    -> downsample to TR by averaging over TR-length windows -> z-score.
    """
    picks = [ch for ch in OCCIPITAL_CHS if ch in raw.ch_names]
    if not picks:
        raise ValueError(
            f"None of {OCCIPITAL_CHS} found in raw (have first few: "
            f"{raw.ch_names[:8]}...)"
        )
    log.info("  occipital channels used: %s", picks)
    raw_a = raw.copy().pick(picks).filter(
        l_freq=ALPHA_BAND[0], h_freq=ALPHA_BAND[1], verbose=False,
    )
    data = raw_a.get_data()  # (n_ch, n_samples)
    env = np.abs(hilbert(data, axis=1))
    log_power = np.log10(env ** 2 + 1e-12)
    avg = log_power.mean(axis=0)

    sfreq = raw_a.info["sfreq"]
    win = max(int(SMOOTH_WIN_S * sfreq), 1)
    smoothed = np.convolve(avg, np.ones(win) / win, mode="same")

    # Downsample to TR by averaging windows of TR seconds.
    samples_per_tr = sfreq * target_tr_s
    n_trs = int(np.floor(len(smoothed) / samples_per_tr))
    out = np.empty(n_trs, dtype=float)
    for t in range(n_trs):
        i = int(round(t * samples_per_tr))
        j = int(round((t + 1) * samples_per_tr))
        out[t] = smoothed[i:j].mean()

    out = (out - out.mean()) / (out.std() + 1e-9)
    log.info("  alpha envelope: %d TRs, mean=%.3f, std=%.3f",
             n_trs, float(out.mean()), float(out.std()))
    return out


def build_design_matrix(alpha_env: np.ndarray, fd: np.ndarray) -> pd.DataFrame:
    """Design matrix: HRF-convolved alpha + motion + cosine drift."""
    n = len(alpha_env)
    frame_times = np.arange(n) * TR

    exp_condition = np.vstack([
        frame_times,
        np.full(n, TR),
        alpha_env,
    ])
    alpha_hrf, _ = compute_regressor(
        exp_condition, "glover", frame_times, con_id="alpha",
    )
    fd_d = np.concatenate([[0.0], np.diff(fd)])
    add_regs = np.hstack([alpha_hrf, fd.reshape(-1, 1), fd_d.reshape(-1, 1)])
    add_reg_names = ["alpha_hrf", "fd", "fd_d"]

    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=None,
        hrf_model=None,         # alpha already convolved above
        drift_model="cosine",
        high_pass=0.01,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
    )
    log.info("  design matrix shape=%s, regressors=%s",
             dm.shape, list(dm.columns))
    return dm


def run_subject(sub: str, bids_root: Path, out_dir: Path,
                stimloc_mask_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("Subject sub-%s", sub)

    bold_path = bids_root / f"sub-{sub}" / "ses-02" / "func" / \
                f"sub-{sub}_ses-02_task-rest.nii.gz"
    if not bold_path.exists():
        log.warning("  missing rest BOLD")
        return {"subject": sub, "error": "missing rest BOLD"}

    bold_img = nib.load(str(bold_path))
    n_trs = bold_img.shape[3]
    log.info("  BOLD shape=%s", bold_img.shape)
    bold_img, fd = motion_correct(bold_path, n_trs)

    raw = load_rest_eeg(bids_root, sub)
    if raw is None:
        return {"subject": sub, "error": "no rest EEG"}

    alpha_env = occipital_alpha_envelope(raw, target_tr_s=TR)

    # Align lengths: trim both to the shorter one.
    n = min(len(alpha_env), n_trs)
    if n < n_trs:
        log.info("  trimming BOLD %d -> %d TRs to match EEG", n_trs, n)
        bold_data = bold_img.get_fdata()[..., :n]
        bold_img = new_img_like(bold_img, bold_data)
    alpha_env = alpha_env[:n]
    fd = fd[:n]

    dm = build_design_matrix(alpha_env, fd)
    mask_img = compute_epi_mask(bold_img)
    n_brain = int(mask_img.get_fdata().astype(bool).sum())

    log.info("Fitting FirstLevelModel")
    glm = FirstLevelModel(
        t_r=TR, mask_img=mask_img, smoothing_fwhm=6.0,
        minimize_memory=False, verbose=0,
    )
    glm.fit(bold_img, design_matrices=dm)
    z_map = glm.compute_contrast("alpha_hrf", output_type="z_score")
    z_data = z_map.get_fdata()
    log.info("  z-map range: [%.2f, %.2f]",
             float(z_data.min()), float(z_data.max()))

    # Restrict to brain mask for whole-brain stats.
    brain_bool = mask_img.get_fdata().astype(bool)
    z_brain = z_data[brain_bool]
    pct_neg = float((z_brain < -2.0).mean() * 100.0)
    pct_pos = float((z_brain > 2.0).mean() * 100.0)

    # Visual-cortex stats (the headline test).
    v1_mask_path = stimloc_mask_dir / f"sub-{sub}" / \
                   f"sub-{sub}_ses-02_stimloc-visual-mask.nii.gz"
    z_mean_v1 = None
    z_median_v1 = None
    n_v1 = None
    pct_neg_v1 = None
    if v1_mask_path.exists():
        v1_img = nib.load(str(v1_mask_path))
        v1_resampled = resample_to_img(
            v1_img, bold_img, interpolation="nearest",
            force_resample=True, copy_header=True,
        )
        v1_bool = v1_resampled.get_fdata().astype(bool) & brain_bool
        n_v1 = int(v1_bool.sum())
        if n_v1 > 0:
            v1_vals = z_data[v1_bool]
            z_mean_v1 = float(v1_vals.mean())
            z_median_v1 = float(np.median(v1_vals))
            pct_neg_v1 = float((v1_vals < 0).mean() * 100.0)
            log.info("  V1 mask: %d voxels, mean z = %.3f, "
                     "median z = %.3f, %% negative = %.1f",
                     n_v1, z_mean_v1, z_median_v1, pct_neg_v1)

    sub_out = out_dir / f"sub-{sub}"
    sub_out.mkdir(parents=True, exist_ok=True)
    zmap_path = sub_out / f"sub-{sub}_alpha-bold-zmap.nii.gz"
    nib.save(z_map, str(zmap_path))
    log.info("  saved %s", zmap_path)

    return {
        "subject": sub,
        "n_trs": int(n),
        "n_brain_voxels": n_brain,
        "fd_mean_mm": float(fd.mean()),
        "fd_max_mm": float(fd.max()),
        "z_min": float(z_data.min()),
        "z_max": float(z_data.max()),
        "pct_neg_brain": pct_neg,
        "pct_pos_brain": pct_pos,
        "n_v1_voxels": n_v1,
        "z_mean_v1": z_mean_v1,
        "z_median_v1": z_median_v1,
        "pct_negative_v1": pct_neg_v1,
        "zmap_path": str(zmap_path),
    }


def discover_subjects(bids_root: Path) -> list[str]:
    subs = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if (sub_dir / "ses-02" / "func" /
                f"{sub_dir.name}_ses-02_task-rest.nii.gz").exists():
            subs.append(sub_dir.name.replace("sub-", ""))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data", type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/fmri_alpha_bold",
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
    (args.output_dir / "alpha_bold_results.json").write_text(
        json.dumps({"per_subject": results}, indent=2)
    )

    print("\n" + "=" * 70)
    print(f"{'Subject':<12} {'TRs':>5} {'FD':>8} {'V1 vox':>7} "
          f"{'mean z[V1]':>11} {'%neg V1':>9}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['subject']:<12} ERROR: {r['error']}")
            continue
        zm = r.get("z_mean_v1")
        pn = r.get("pct_negative_v1")
        zm_str = f"{zm:.3f}" if zm is not None else "n/a"
        pn_str = f"{pn:.1f}%" if pn is not None else "n/a"
        print(f"{r['subject']:<12} {r['n_trs']:>5} "
              f"{r['fd_mean_mm']:>7.3f}mm {r['n_v1_voxels'] or 0:>7} "
              f"{zm_str:>11} {pn_str:>9}")
    print("=" * 70)
    print("Interpretation:")
    print("  Canonical: negative z[V1] (high alpha -> low BOLD).")
    print("  Reversed:  positive z[V1] -- the cross-modal anomaly signature.")


if __name__ == "__main__":
    main()
