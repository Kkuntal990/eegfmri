"""Per-task alpha-BOLD coupling GLM (Goldman 2002 / Ingram 2024 style).

Generalises ``fmri_eeg_alpha_glm.py`` to fit the alpha-BOLD coupling map
for each of {rest, swm_run-1, swm_run-2} on the same subject. The output
is one z-map per (subject, task), all in native space, with the per-subject
stimloc V1 mask used as the headline ROI.

This is the Week-1 deliverable of the Gap-#4 reliability sprint: three
within-session alpha-BOLD coupling maps per subject -> ICC(3,1) across
them in V1 in Week 2.

Output layout:

  results/alpha_bold_per_task/sub-XXX/
    task-rest/sub-XXX_task-rest_alpha-bold-zmap.nii.gz
    task-rest/sub-XXX_task-rest_summary.json
    task-swm_run-1/...
    task-swm_run-2/...
  results/alpha_bold_per_task/per_task_results.json   # all subjects/tasks

Usage:
    python scripts/fmri_alpha_bold_per_task.py \
        --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/alpha_bold_per_task \
        --subjects 500 1070302 \
        --tasks rest swm_run-1 swm_run-2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cache import get_cleaned_eeg_for_task, get_motion_corrected_bold  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TR = 1.0
ALPHA_BAND = (8.0, 12.0)
SMOOTH_WIN_S = 5.0
# Occipital + parietal channels in the 128-ch EGI HydroCel cap. Same
# channels used in validate_eeg.py for the Berger contrast.
OCCIPITAL_CHS = ["E62", "E67", "E70", "E72", "E75", "E76", "E83", "E84"]

SUPPORTED_TASKS = ("rest", "swm_run-1", "swm_run-2")


def parse_task(task: str) -> tuple[str, str | None, str]:
    """Map our ``--task`` value to (eeg_task, run_id, bold_filename_suffix).

    The BIDS file naming on disk is::

        rest        -> sub-X_ses-02_task-rest.nii.gz                eeg task='rest', run=None
        swm_run-1   -> sub-X_ses-02_task-swm_run-1.nii.gz           eeg task='swm',  run='1'
        swm_run-2   -> sub-X_ses-02_task-swm_run-2.nii.gz           eeg task='swm',  run='2'
    """
    if task == "rest":
        return "rest", None, "rest"
    if task in ("swm_run-1", "swm_run-2"):
        run_id = task.split("-")[-1]
        return "swm", run_id, f"swm_run-{run_id}"
    raise ValueError(f"unsupported task {task!r}; expected one of {SUPPORTED_TASKS}")


def load_eeg_run(bids_root: Path, sub: str, eeg_task: str, run_id: str | None):
    """Return the cleaned MNE Raw for one (subject, task, run)."""
    raws_by_run = get_cleaned_eeg_for_task(bids_root, sub, ses="02", task=eeg_task)
    if not raws_by_run:
        return None
    if run_id is None:
        # Single-run task; cache returns {None: raw} OR the first run.
        return next(iter(raws_by_run.values()))
    # Multi-run: cache keys are str run ids.
    if run_id in raws_by_run:
        return raws_by_run[run_id]
    log.warning("  run %s missing from cleaned EEG (have %s)",
                run_id, list(raws_by_run.keys()))
    return None


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
        hrf_model=None,
        drift_model="cosine",
        high_pass=0.01,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
    )
    log.info("  design matrix shape=%s, regressors=%s",
             dm.shape, list(dm.columns))
    return dm


def run_subject_task(sub: str, task: str, bids_root: Path, out_dir: Path,
                     stimloc_mask_dir: Path) -> dict:
    log.info("=" * 70)
    log.info("Subject sub-%s | task %s", sub, task)

    eeg_task, run_id, bold_suffix = parse_task(task)
    bold_path = bids_root / f"sub-{sub}" / "ses-02" / "func" / \
                f"sub-{sub}_ses-02_task-{bold_suffix}.nii.gz"
    if not bold_path.exists():
        log.warning("  missing BOLD: %s", bold_path)
        return {"subject": sub, "task": task, "error": f"missing BOLD {bold_path.name}"}

    bold_img, fd = get_motion_corrected_bold(
        bold_path, sub, ses="02", task=eeg_task, run=run_id,
    )
    n_trs = bold_img.shape[3]
    log.info("  BOLD shape=%s, FD mean=%.3f mm", bold_img.shape,
             float(fd.mean()))

    raw = load_eeg_run(bids_root, sub, eeg_task, run_id)
    if raw is None:
        return {"subject": sub, "task": task, "error": "no EEG run"}

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

    brain_bool = mask_img.get_fdata().astype(bool)
    z_brain = z_data[brain_bool]
    pct_neg = float((z_brain < -2.0).mean() * 100.0)
    pct_pos = float((z_brain > 2.0).mean() * 100.0)

    # V1 stats from per-subject stimloc mask.
    v1_mask_path = stimloc_mask_dir / f"sub-{sub}" / \
                   f"sub-{sub}_ses-02_stimloc-visual-mask.nii.gz"
    z_mean_v1 = z_median_v1 = pct_neg_v1 = None
    n_v1 = None
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

    sub_task_dir = out_dir / f"sub-{sub}" / f"task-{task}"
    sub_task_dir.mkdir(parents=True, exist_ok=True)
    zmap_path = sub_task_dir / f"sub-{sub}_task-{task}_alpha-bold-zmap.nii.gz"
    nib.save(z_map, str(zmap_path))
    log.info("  saved %s", zmap_path)

    summary = {
        "subject": sub,
        "task": task,
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
    (sub_task_dir / f"sub-{sub}_task-{task}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def discover_subjects(bids_root: Path, tasks: list[str]) -> list[str]:
    """Subjects that have all requested tasks present."""
    subs = []
    for sub_dir in sorted(bids_root.glob("sub-*")):
        ok = True
        for t in tasks:
            _, _, bold_suffix = parse_task(t)
            f = sub_dir / "ses-02" / "func" / \
                f"{sub_dir.name}_ses-02_task-{bold_suffix}.nii.gz"
            if not f.exists():
                ok = False
                break
        if ok:
            subs.append(sub_dir.name.replace("sub-", ""))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data", type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/alpha_bold_per_task",
                    type=Path)
    ap.add_argument("--stimloc-mask-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=list(SUPPORTED_TASKS),
                    choices=list(SUPPORTED_TASKS))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subs = args.subjects or discover_subjects(args.bids_root, args.tasks)
    log.info("Subjects: %s", subs)
    log.info("Tasks:    %s", args.tasks)

    results = []
    for s in subs:
        for t in args.tasks:
            try:
                results.append(run_subject_task(
                    s, t, args.bids_root, args.output_dir, args.stimloc_mask_dir,
                ))
            except Exception as e:  # noqa: BLE001 - one failed task shouldn't kill the batch
                log.exception("FAILED sub-%s task=%s", s, t)
                results.append({"subject": s, "task": t, "error": str(e)})

    (args.output_dir / "per_task_results.json").write_text(
        json.dumps({"per_subject_task": results}, indent=2)
    )

    print("\n" + "=" * 80)
    print(f"{'Subject':<10} {'Task':<14} {'TRs':>5} {'FD':>8} "
          f"{'V1 vox':>7} {'mean z[V1]':>11} {'%neg V1':>9}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['subject']:<10} {r['task']:<14} ERROR: {r['error']}")
            continue
        zm = r.get("z_mean_v1")
        pn = r.get("pct_negative_v1")
        zm_str = f"{zm:.3f}" if zm is not None else "n/a"
        pn_str = f"{pn:.1f}%" if pn is not None else "n/a"
        print(f"{r['subject']:<10} {r['task']:<14} {r['n_trs']:>5} "
              f"{r['fd_mean_mm']:>7.3f}mm {r['n_v1_voxels'] or 0:>7} "
              f"{zm_str:>11} {pn_str:>9}")
    print("=" * 80)
    print("Interpretation:")
    print("  Canonical: negative z[V1] across all tasks (high alpha -> low BOLD).")
    print("  Phenotype: subjects with consistent z[V1] sign + magnitude across the")
    print("             three tasks are candidates for high-ICC alpha-BOLD coupling.")


if __name__ == "__main__":
    main()
