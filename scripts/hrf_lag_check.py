"""HRF-lag consistency check for the per-task alpha-BOLD coupling.

For each (subject, task), compute the V1-mean BOLD time-series (mean over
the per-subject stimloc V1 mask) and the occipital alpha envelope at TR
resolution, then cross-correlate over lags [-2, +12] s. Report the lag at
which the correlation is most negative (canonical alpha-BOLD coupling) and
the lag at which |r| is maximum.

Why we need it: voxelwise ICC of alpha-BOLD coupling came out essentially
zero across rest + swm_run-1 + swm_run-2 (Week-2 result). A reviewer-level
question is whether the HRF time-to-peak drifts across tasks -- if so, a
fixed-Glover GLM would mis-align the alpha regressor by 1-2 TRs and
artificially deflate cross-task reliability even at the ROI level. This
script tells us whether the HRF lag is stable enough to trust the fixed
Glover convolution.

Output:
  results/alpha_bold_hrf_lag/sub-XXX_task-YYY.json
  results/alpha_bold_hrf_lag/summary.json
  results/alpha_bold_hrf_lag/summary.csv

Usage:
    python scripts/hrf_lag_check.py \
        --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
        --stimloc-mask-dir /projects/bbnv/kkokate/eegfmri/results/stimloc_mask \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/alpha_bold_hrf_lag
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import new_img_like, resample_to_img
from nilearn.masking import compute_epi_mask
from scipy.signal import butter, filtfilt, hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cache import get_cleaned_eeg_for_task, get_motion_corrected_bold  # noqa: E402
from fmri_alpha_bold_per_task import (  # noqa: E402
    OCCIPITAL_CHS, parse_task, SUPPORTED_TASKS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TR = 1.0
ALPHA_BAND = (8.0, 12.0)
SMOOTH_WIN_S = 5.0
LAG_MIN_TR = -2
LAG_MAX_TR = 12
HIGHPASS_HZ = 0.01


def occipital_alpha_envelope(raw, target_tr_s: float = TR) -> np.ndarray:
    """Same routine as fmri_alpha_bold_per_task.py, kept here as a local copy
    so changes to the lag-check filtering chain don't accidentally affect the
    GLM script.
    """
    picks = [ch for ch in OCCIPITAL_CHS if ch in raw.ch_names]
    if not picks:
        raise ValueError(f"no occipital channels found in raw")
    raw_a = raw.copy().pick(picks).filter(
        l_freq=ALPHA_BAND[0], h_freq=ALPHA_BAND[1], verbose=False,
    )
    data = raw_a.get_data()
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
    return out


def highpass(ts: np.ndarray, cutoff_hz: float = HIGHPASS_HZ,
             tr_s: float = TR) -> np.ndarray:
    """2nd-order Butterworth high-pass at cutoff_hz, matched to GLM high_pass."""
    fs = 1.0 / tr_s
    b, a = butter(2, cutoff_hz / (fs / 2.0), btype="high")
    return filtfilt(b, a, ts)


def lagged_pearson(x: np.ndarray, y: np.ndarray,
                   lag_min: int, lag_max: int) -> dict[int, float]:
    """Correlate x with y(t+lag): positive lag means y lags x."""
    out = {}
    for k in range(lag_min, lag_max + 1):
        if k >= 0:
            x_ = x[: len(x) - k]
            y_ = y[k:]
        else:
            x_ = x[-k:]
            y_ = y[: len(y) + k]
        if x_.std() > 0 and y_.std() > 0:
            out[k] = float(np.corrcoef(x_, y_)[0, 1])
        else:
            out[k] = float("nan")
    return out


def run_one(sub: str, task: str, bids_root: Path,
            stimloc_mask_dir: Path, out_dir: Path) -> dict:
    log.info("=" * 60)
    log.info("sub-%s | task %s", sub, task)
    eeg_task, run_id, bold_suffix = parse_task(task)
    bold_path = bids_root / f"sub-{sub}" / "ses-02" / "func" / \
                f"sub-{sub}_ses-02_task-{bold_suffix}.nii.gz"
    if not bold_path.exists():
        return {"subject": sub, "task": task, "error": "missing BOLD"}

    bold_img, _fd = get_motion_corrected_bold(
        bold_path, sub, ses="02", task=eeg_task, run=run_id,
    )

    v1_path = stimloc_mask_dir / f"sub-{sub}" / \
              f"sub-{sub}_ses-02_stimloc-visual-mask.nii.gz"
    if not v1_path.exists():
        return {"subject": sub, "task": task, "error": "no V1 mask"}
    v1_img = nib.load(str(v1_path))
    v1_ref = resample_to_img(v1_img, bold_img, interpolation="nearest",
                             force_resample=True, copy_header=True)
    mask_img = compute_epi_mask(bold_img)
    v1_bool = v1_ref.get_fdata().astype(bool) & mask_img.get_fdata().astype(bool)
    n_v1 = int(v1_bool.sum())
    if n_v1 < 10:
        return {"subject": sub, "task": task, "error": "V1 mask too small"}

    bold_data = bold_img.get_fdata()
    v1_ts = bold_data[v1_bool, :].mean(axis=0)
    v1_ts = highpass(v1_ts)
    v1_ts = (v1_ts - v1_ts.mean()) / (v1_ts.std() + 1e-9)
    log.info("  V1 BOLD time-series: %d TRs (V1 vox=%d)", len(v1_ts), n_v1)

    raws = get_cleaned_eeg_for_task(bids_root, sub, ses="02", task=eeg_task)
    raw = raws.get(run_id) if run_id is not None else next(iter(raws.values()))
    if raw is None:
        return {"subject": sub, "task": task, "error": "no EEG"}
    alpha_env = occipital_alpha_envelope(raw)

    n = min(len(alpha_env), len(v1_ts))
    alpha_env = alpha_env[:n]
    v1_ts = v1_ts[:n]

    r_by_lag = lagged_pearson(alpha_env, v1_ts, LAG_MIN_TR, LAG_MAX_TR)
    lags = sorted(r_by_lag.keys())
    rs = np.array([r_by_lag[k] for k in lags])

    lag_min_r = lags[int(np.nanargmin(rs))]
    r_at_lag_min = float(rs[int(np.nanargmin(rs))])
    lag_max_abs = lags[int(np.nanargmax(np.abs(rs)))]
    r_at_lag_max_abs = float(rs[int(np.nanargmax(np.abs(rs)))])

    log.info("  most-negative r = %.3f at lag = %d TR", r_at_lag_min, lag_min_r)
    log.info("  max |r|         = %.3f at lag = %d TR",
             abs(r_at_lag_max_abs), lag_max_abs)

    summary = {
        "subject": sub,
        "task": task,
        "n_trs": int(n),
        "n_v1_voxels": n_v1,
        "lag_tr_of_min_r": int(lag_min_r),
        "r_at_lag_min": r_at_lag_min,
        "lag_tr_of_max_abs_r": int(lag_max_abs),
        "r_at_lag_max_abs": r_at_lag_max_abs,
        "r_by_lag": {str(k): float(r_by_lag[k]) for k in lags},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sub-{sub}_task-{task}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data", type=Path)
    ap.add_argument("--stimloc-mask-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/alpha_bold_hrf_lag",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=["500", "1070302"])
    ap.add_argument("--tasks", nargs="*", default=list(SUPPORTED_TASKS),
                    choices=list(SUPPORTED_TASKS))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for s in args.subjects:
        for t in args.tasks:
            try:
                results.append(run_one(
                    s, t, args.bids_root, args.stimloc_mask_dir, args.output_dir,
                ))
            except Exception as e:  # noqa: BLE001
                log.exception("FAILED sub-%s task=%s", s, t)
                results.append({"subject": s, "task": t, "error": str(e)})

    (args.output_dir / "summary.json").write_text(
        json.dumps({"per_subject_task": results}, indent=2)
    )

    rows = [{k: v for k, v in r.items() if k != "r_by_lag"} for r in results]
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)

    print("\n" + "=" * 86)
    print(f"{'Subject':<10} {'Task':<14} {'TRs':>5} {'lag(min r) [TR]':>16} "
          f"{'r at lag':>10} {'|r|max@lag':>11}")
    print("-" * 86)
    for r in results:
        if "error" in r:
            print(f"{r['subject']:<10} {r['task']:<14} ERROR {r['error']}")
            continue
        print(f"{r['subject']:<10} {r['task']:<14} {r['n_trs']:>5} "
              f"{r['lag_tr_of_min_r']:>16} "
              f"{r['r_at_lag_min']:>10.3f} "
              f"{r['lag_tr_of_max_abs_r']:>5} ({r['r_at_lag_max_abs']:+.3f})")
    print("=" * 86)
    print("Reading the result:")
    print("  Canonical HRF time-to-peak with alpha-BOLD coupling: 4-6 TR (most negative r).")
    print("  If lag is consistent across rest/swm-1/swm-2 within a subject, the fixed-Glover")
    print("  GLM is well-aligned; voxelwise ICC ≈ 0 cannot be blamed on lag drift.")
    print("  If lag drifts > 2 TR between tasks, refit GLM with subject-specific HRF.")


if __name__ == "__main__":
    main()
