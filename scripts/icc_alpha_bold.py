"""Within-subject reliability of alpha-BOLD coupling maps across tasks.

Reads the three per-task z-maps produced by ``fmri_alpha_bold_per_task.py``
(rest, swm_run-1, swm_run-2), restricts to each subject's stimloc-derived
V1 mask in native space, and computes:

  * ICC(3,1)   -- Shrout & Fleiss 1979, two-way mixed, single rater,
                  consistency. Voxels = "subjects", tasks = "raters".
  * Pairwise Pearson r across the three task pairs.
  * Dice overlap of supra-threshold voxels (|z| > 2.3) for each pair.
  * Mean / median z per task within V1 (sanity check vs canonical sign).

Output:
  results/alpha_bold_reliability/sub-XXX.json
  results/alpha_bold_reliability/group_summary.json
  results/alpha_bold_reliability/group_summary.csv

Usage:
    python scripts/icc_alpha_bold.py \
        --zmap-dir /projects/bbnv/kkokate/eegfmri/results/alpha_bold_per_task \
        --stimloc-mask-dir /projects/bbnv/kkokate/eegfmri/results/stimloc_mask \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/alpha_bold_reliability
"""

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TASKS = ("rest", "swm_run-1", "swm_run-2")
SUPRA_THRESH = 2.3  # |z| > 2.3 for Dice overlap


def icc_3_1(matrix: np.ndarray) -> float:
    """Shrout & Fleiss (1979) ICC(3,1) -- two-way mixed, single rater, consistency.

    matrix shape: (n_targets, n_raters). Here targets = voxels, raters = tasks.

    ICC(3,1) = (MSR - MSE) / (MSR + (k-1)*MSE)

    where:
      MSR = mean square between targets (voxels)
      MSE = residual mean square
      k   = number of raters (tasks)
    """
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand_mean = matrix.mean()
    target_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    ss_total = ((matrix - grand_mean) ** 2).sum()
    ss_target = k * ((target_means - grand_mean) ** 2).sum()
    ss_rater = n * ((rater_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_target - ss_rater
    df_target = n - 1
    df_rater = k - 1
    df_error = (n - 1) * (k - 1)
    ms_target = ss_target / df_target if df_target else float("nan")
    ms_error = ss_error / df_error if df_error else float("nan")
    if not np.isfinite(ms_target) or not np.isfinite(ms_error) or ms_error <= 0:
        return float("nan")
    return float((ms_target - ms_error) / (ms_target + (k - 1) * ms_error))


def load_zmap(zmap_dir: Path, sub: str, task: str) -> Path | None:
    p = zmap_dir / f"sub-{sub}" / f"task-{task}" / \
        f"sub-{sub}_task-{task}_alpha-bold-zmap.nii.gz"
    return p if p.exists() else None


def discover_subjects(zmap_dir: Path) -> list[str]:
    subs = []
    for d in sorted(zmap_dir.glob("sub-*")):
        sub = d.name.replace("sub-", "")
        if all(load_zmap(zmap_dir, sub, t) for t in TASKS):
            subs.append(sub)
    return subs


def reliability_for_subject(sub: str, zmap_dir: Path,
                            stimloc_mask_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("Subject sub-%s", sub)

    paths = {t: load_zmap(zmap_dir, sub, t) for t in TASKS}
    missing = [t for t, p in paths.items() if p is None]
    if missing:
        return {"subject": sub, "error": f"missing z-maps: {missing}"}

    imgs = {t: nib.load(str(p)) for t, p in paths.items()}
    ref = imgs[TASKS[0]]  # reference space (rest)

    # V1 mask in subject native space; resample to the reference grid.
    v1_path = stimloc_mask_dir / f"sub-{sub}" / \
              f"sub-{sub}_ses-02_stimloc-visual-mask.nii.gz"
    if not v1_path.exists():
        return {"subject": sub, "error": "no stimloc V1 mask"}
    v1_img = nib.load(str(v1_path))
    v1_ref = resample_to_img(v1_img, ref, interpolation="nearest",
                             force_resample=True, copy_header=True)
    v1_bool = v1_ref.get_fdata().astype(bool)
    n_v1 = int(v1_bool.sum())
    log.info("  V1 voxels: %d", n_v1)
    if n_v1 < 10:
        return {"subject": sub, "n_v1_voxels": n_v1,
                "error": "V1 mask too small"}

    # Resample every other task's z-map to the reference grid so voxels line up.
    z_by_task = {}
    for t, img in imgs.items():
        if t == TASKS[0]:
            z_by_task[t] = img.get_fdata()
        else:
            z_by_task[t] = resample_to_img(
                img, ref, interpolation="continuous",
                force_resample=True, copy_header=True,
            ).get_fdata()

    # Build (n_voxels, n_tasks) matrix.
    Z = np.column_stack([z_by_task[t][v1_bool] for t in TASKS])
    log.info("  Z matrix: %s, finite=%d", Z.shape, int(np.isfinite(Z).all(axis=1).sum()))

    finite = np.isfinite(Z).all(axis=1)
    Z = Z[finite]
    n_used = Z.shape[0]
    if n_used < 10:
        return {"subject": sub, "n_v1_voxels": n_v1, "n_used": n_used,
                "error": "insufficient finite voxels"}

    icc = icc_3_1(Z)
    log.info("  ICC(3,1) = %.4f", icc)

    pearson = {}
    dice = {}
    for (i, t1), (j, t2) in combinations(enumerate(TASKS), 2):
        z1, z2 = Z[:, i], Z[:, j]
        if z1.std() > 0 and z2.std() > 0:
            r = float(np.corrcoef(z1, z2)[0, 1])
        else:
            r = float("nan")
        pearson[f"{t1}_vs_{t2}"] = r
        b1 = np.abs(z1) > SUPRA_THRESH
        b2 = np.abs(z2) > SUPRA_THRESH
        denom = (b1.sum() + b2.sum())
        dice[f"{t1}_vs_{t2}"] = (
            float(2.0 * (b1 & b2).sum() / denom) if denom > 0 else float("nan")
        )
        log.info("    %-22s pearson=%+.3f  dice=%.3f",
                 f"{t1} vs {t2}", r, dice[f"{t1}_vs_{t2}"])

    per_task = {}
    for i, t in enumerate(TASKS):
        v = Z[:, i]
        per_task[t] = {
            "mean_z_v1": float(v.mean()),
            "median_z_v1": float(np.median(v)),
            "pct_negative_v1": float((v < 0).mean() * 100.0),
            "pct_supra_v1": float((np.abs(v) > SUPRA_THRESH).mean() * 100.0),
        }

    return {
        "subject": sub,
        "n_v1_voxels": n_v1,
        "n_used_voxels": int(n_used),
        "icc_3_1_v1": icc,
        "pearson_v1": pearson,
        "dice_v1_supra": dice,
        "per_task": per_task,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmap-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/alpha_bold_per_task",
                    type=Path)
    ap.add_argument("--stimloc-mask-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/alpha_bold_reliability",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subs = args.subjects or discover_subjects(args.zmap_dir)
    log.info("Subjects: %s", subs)

    rows = []
    for s in subs:
        r = reliability_for_subject(s, args.zmap_dir, args.stimloc_mask_dir)
        (args.output_dir / f"sub-{s}.json").write_text(json.dumps(r, indent=2))
        rows.append(r)

    (args.output_dir / "group_summary.json").write_text(
        json.dumps({"per_subject": rows}, indent=2)
    )

    # Flat CSV for easy inspection.
    flat = []
    for r in rows:
        if "error" in r:
            flat.append({"subject": r["subject"], "error": r["error"]})
            continue
        row = {
            "subject": r["subject"],
            "n_v1_voxels": r["n_v1_voxels"],
            "icc_3_1_v1": r["icc_3_1_v1"],
        }
        for k, v in r["pearson_v1"].items():
            row[f"pearson_{k}"] = v
        for k, v in r["dice_v1_supra"].items():
            row[f"dice_{k}"] = v
        for t, d in r["per_task"].items():
            row[f"mean_z_v1_{t}"] = d["mean_z_v1"]
            row[f"pct_neg_v1_{t}"] = d["pct_negative_v1"]
        flat.append(row)
    pd.DataFrame(flat).to_csv(args.output_dir / "group_summary.csv", index=False)

    print("\n" + "=" * 90)
    print(f"{'Subject':<10} {'V1 vox':>7} {'ICC(3,1)':>10} "
          f"{'r rest~swm1':>13} {'r rest~swm2':>13} {'r swm1~swm2':>13}")
    print("-" * 90)
    for r in rows:
        if "error" in r:
            print(f"{r['subject']:<10} ERROR: {r['error']}")
            continue
        p = r["pearson_v1"]
        print(f"{r['subject']:<10} {r['n_v1_voxels']:>7} "
              f"{r['icc_3_1_v1']:>10.4f} "
              f"{p['rest_vs_swm_run-1']:>13.3f} "
              f"{p['rest_vs_swm_run-2']:>13.3f} "
              f"{p['swm_run-1_vs_swm_run-2']:>13.3f}")
    print("=" * 90)
    print("Prediction:")
    print("  Canonical-coupling subject (sub-500):    ICC > 0.3 + all-negative mean z[V1].")
    print("  Drowsy/reversed subject  (sub-1070302):  ICC near zero + mixed-sign z[V1].")


if __name__ == "__main__":
    main()
