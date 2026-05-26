"""Within-run sliding-window alpha-BOLD coupling dynamics.

For each (subject, task), slide a window through the run and recompute the
Pearson correlation between the occipital alpha envelope and V1-mean BOLD
at the canonical Glover lag (+5 TR). This gives a time-course of coupling
strength inside each task.

Outputs per-window r-trajectories, summary stats (mean, SD, min, max),
and -- for the two swm runs -- correlations with trial-level behaviour
(mean RT and accuracy of Sternberg trials whose target onset falls inside
each window).

Hypothesis after HRF-lag check (job 18476644):
  sub-500       -> low within-run SD across all three tasks (stable coupling)
  sub-1070302   -> within-task collapse on swm; ideally rest stays canonical
                   and swm runs swing through zero

Output:
  results/alpha_bold_within_run/sub-XXX_task-YYY.json     # per-window r + behaviour
  results/alpha_bold_within_run/summary.csv               # one row per (sub, task)
  results/alpha_bold_within_run/summary.json

Usage:
    python scripts/alpha_bold_within_run.py \
        --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
        --stimloc-mask-dir /projects/bbnv/kkokate/eegfmri/results/stimloc_mask \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/alpha_bold_within_run
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
HIGHPASS_HZ = 0.01

# Sliding-window choices.
WIN_TR = 50           # ~50 s window (3-4 Sternberg trials)
STEP_TR = 25          # 50% overlap for a smoother trajectory
HRF_LAG_TR = 5        # Glover canonical peak; matches the lag found in
                      # sub-500 across all tasks and sub-1070302 at rest.


def occipital_alpha_envelope(raw) -> np.ndarray:
    picks = [ch for ch in OCCIPITAL_CHS if ch in raw.ch_names]
    if not picks:
        raise ValueError("no occipital channels in raw")
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
    samples_per_tr = sfreq * TR
    n_trs = int(np.floor(len(smoothed) / samples_per_tr))
    out = np.empty(n_trs, dtype=float)
    for t in range(n_trs):
        i = int(round(t * samples_per_tr))
        j = int(round((t + 1) * samples_per_tr))
        out[t] = smoothed[i:j].mean()
    return (out - out.mean()) / (out.std() + 1e-9)


def highpass(ts: np.ndarray) -> np.ndarray:
    fs = 1.0 / TR
    b, a = butter(2, HIGHPASS_HZ / (fs / 2.0), btype="high")
    return filtfilt(b, a, ts)


def load_swm_trials(events_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(events_tsv, sep="\t")
    val = df["value"].astype(str)
    targ = df[val.str.startswith("SDRT_targ_")].copy()
    if targ.empty:
        return targ
    targ["onset_s"] = pd.to_numeric(targ["onsetRelToMRIstart"], errors="coerce")
    if "RT" in targ.columns:
        targ["rt"] = pd.to_numeric(targ["RT"], errors="coerce")
    elif "responseTime" in targ.columns:
        targ["rt"] = pd.to_numeric(targ["responseTime"], errors="coerce")
    else:
        targ["rt"] = np.nan
    if "accuracy" in targ.columns:
        targ["acc"] = pd.to_numeric(targ["accuracy"], errors="coerce")
    else:
        targ["acc"] = np.nan
    return targ[["onset_s", "rt", "acc"]].dropna(subset=["onset_s"])


def sliding_windows(n: int, win: int, step: int) -> list[tuple[int, int]]:
    out = []
    s = 0
    while s + win <= n:
        out.append((s, s + win))
        s += step
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
    bold_data = bold_img.get_fdata()
    v1_ts = bold_data[v1_bool, :].mean(axis=0)
    v1_ts = highpass(v1_ts)
    v1_ts = (v1_ts - v1_ts.mean()) / (v1_ts.std() + 1e-9)

    raws = get_cleaned_eeg_for_task(bids_root, sub, ses="02", task=eeg_task)
    raw = raws.get(run_id) if run_id is not None else next(iter(raws.values()))
    if raw is None:
        return {"subject": sub, "task": task, "error": "no EEG"}
    alpha_env = occipital_alpha_envelope(raw)

    n = min(len(alpha_env), len(v1_ts))
    alpha_env = alpha_env[:n]
    v1_ts = v1_ts[:n]

    # Apply the fixed Glover lag: correlate alpha[t] with V1_BOLD[t + HRF_LAG_TR].
    lag = HRF_LAG_TR
    eff_n = n - lag
    if eff_n < WIN_TR:
        return {"subject": sub, "task": task,
                "error": f"too few TRs ({eff_n}) for window {WIN_TR}"}
    a = alpha_env[:eff_n]
    b = v1_ts[lag:lag + eff_n]

    # Optional behaviour for swm runs.
    trials = None
    if eeg_task == "swm":
        events_tsv = bids_root / f"sub-{sub}" / "ses-02" / "eeg" / \
                     f"sub-{sub}_ses-02_task-swm_run-{run_id}_events.tsv"
        if events_tsv.exists():
            trials = load_swm_trials(events_tsv)
            log.info("  loaded %d Sternberg trials", len(trials))

    windows = sliding_windows(eff_n, WIN_TR, STEP_TR)
    log.info("  %d sliding windows (TR=%d, step=%d, eff_n=%d)",
             len(windows), WIN_TR, STEP_TR, eff_n)

    per_window = []
    for w_idx, (s, e) in enumerate(windows):
        a_w = a[s:e]
        b_w = b[s:e]
        if a_w.std() > 0 and b_w.std() > 0:
            r = float(np.corrcoef(a_w, b_w)[0, 1])
        else:
            r = float("nan")
        # Centre time in original TRs (BOLD frame).
        centre_tr = s + WIN_TR / 2.0 + lag  # +lag because BOLD slice is shifted

        row = {"window_idx": w_idx, "start_tr": int(s + lag),
               "end_tr": int(e + lag), "centre_tr": float(centre_tr), "r": r}
        if trials is not None and not trials.empty:
            in_win = trials[
                (trials["onset_s"] >= s + lag) & (trials["onset_s"] < e + lag)
            ]
            row["n_trials"] = int(len(in_win))
            row["rt_mean"] = float(in_win["rt"].mean()) if len(in_win) else float("nan")
            row["acc_mean"] = float(in_win["acc"].mean()) if len(in_win) else float("nan")
        per_window.append(row)

    rs = np.array([w["r"] for w in per_window], dtype=float)
    summary = {
        "subject": sub,
        "task": task,
        "n_v1_voxels": n_v1,
        "n_trs_effective": int(eff_n),
        "lag_used_tr": int(lag),
        "n_windows": len(windows),
        "window_tr": WIN_TR,
        "step_tr": STEP_TR,
        "r_mean": float(np.nanmean(rs)),
        "r_sd": float(np.nanstd(rs)),
        "r_min": float(np.nanmin(rs)),
        "r_max": float(np.nanmax(rs)),
        "pct_windows_negative": float(np.nanmean(rs < 0) * 100.0),
        "per_window": per_window,
    }

    # Behaviour correlations for swm runs.
    if trials is not None and not trials.empty:
        win_rt = np.array([w.get("rt_mean", np.nan) for w in per_window])
        win_acc = np.array([w.get("acc_mean", np.nan) for w in per_window])
        mask_rt = np.isfinite(rs) & np.isfinite(win_rt)
        mask_acc = np.isfinite(rs) & np.isfinite(win_acc)
        summary["r_vs_rt_corr"] = (
            float(np.corrcoef(rs[mask_rt], win_rt[mask_rt])[0, 1])
            if mask_rt.sum() >= 3 else float("nan")
        )
        summary["r_vs_acc_corr"] = (
            float(np.corrcoef(rs[mask_acc], win_acc[mask_acc])[0, 1])
            if mask_acc.sum() >= 3 else float("nan")
        )
        log.info("  r vs RT  (Pearson) = %.3f",
                 summary["r_vs_rt_corr"] if np.isfinite(summary["r_vs_rt_corr"]) else float("nan"))
        log.info("  r vs ACC (Pearson) = %.3f",
                 summary["r_vs_acc_corr"] if np.isfinite(summary["r_vs_acc_corr"]) else float("nan"))

    log.info("  r: mean=%+.3f  SD=%.3f  min=%+.3f  max=%+.3f  %%neg=%.1f",
             summary["r_mean"], summary["r_sd"], summary["r_min"],
             summary["r_max"], summary["pct_windows_negative"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sub-{sub}_task-{task}.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data", type=Path)
    ap.add_argument("--stimloc-mask-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/alpha_bold_within_run",
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
                results.append(run_one(s, t, args.bids_root,
                                       args.stimloc_mask_dir, args.output_dir))
            except Exception as e:  # noqa: BLE001
                log.exception("FAILED sub-%s task=%s", s, t)
                results.append({"subject": s, "task": t, "error": str(e)})

    (args.output_dir / "summary.json").write_text(
        json.dumps({"per_subject_task": results}, indent=2)
    )
    rows = [{k: v for k, v in r.items() if k != "per_window"} for r in results]
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)

    print("\n" + "=" * 100)
    print(f"{'Subject':<10} {'Task':<14} {'#win':>5} {'r mean':>8} {'r SD':>7} "
          f"{'r min':>8} {'r max':>8} {'%neg':>6} {'r~RT':>7} {'r~ACC':>7}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['subject']:<10} {r['task']:<14} ERROR: {r['error']}")
            continue
        rt = r.get("r_vs_rt_corr")
        ac = r.get("r_vs_acc_corr")
        rt_s = f"{rt:+.3f}" if rt is not None and np.isfinite(rt) else "  n/a"
        ac_s = f"{ac:+.3f}" if ac is not None and np.isfinite(ac) else "  n/a"
        print(f"{r['subject']:<10} {r['task']:<14} {r['n_windows']:>5} "
              f"{r['r_mean']:>+.3f} {r['r_sd']:>.3f} "
              f"{r['r_min']:>+.3f} {r['r_max']:>+.3f} "
              f"{r['pct_windows_negative']:>5.1f}% {rt_s:>7} {ac_s:>7}")
    print("=" * 100)
    print("Reading the result:")
    print("  Stable canonical: r mean strongly negative, r SD small, % windows negative high.")
    print("  State-dependent collapse: rest negative, swm runs swing through zero (SD large).")
    print("  Validity (swm only): r ~ RT positive means coupling is weaker when RTs are longer")
    print("                       (drowsy = weaker coupling = slower behaviour).")


if __name__ == "__main__":
    main()
