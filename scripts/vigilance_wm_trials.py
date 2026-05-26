"""Vigilance x working-memory trial-locked analysis on the Sternberg runs.

For each (subject, run, trial) we extract:

  Behaviour:
    - load (1 or 5)
    - accuracy (0 / 1)
    - reactionTime (s)
    - maintenance duration (s)

  EEG features (posterior 8-12 Hz alpha, frontal-midline 4-7 Hz theta):
    - alpha_baseline   : -2.0 to 0.0 s pre-target  (vigilance baseline)
    - alpha_encode     :  0.0 to 2.0 s post-target (desynchronization window)
    - alpha_engaged    :  0.0 to (2 + maintDuration) s post-target
                         (the full WM-active interval; coupling proxy)
    - theta_maintain   :  2.0 to (2 + maintDuration) s post-target
                         (Jensen & Tesche 2002 frontal-theta WM signature)

  fMRI features (per-subject stimloc V1 mask, motion-corrected BOLD):
    - bold_v1_engaged  : mean V1 BOLD across TRs in
                         [target_onset + 5, target_onset + 5 + 2 + maintDuration]
                         (HRF-lagged engaged window)

Five sub-analyses derived from this trial table:

  1. Behaviour: mean ACC / RT per (subject, load), load5-load1 decrement.
  2. Alpha desync: log(alpha_encode / alpha_baseline), per (subject, load).
     Canonical: more negative under load5 (more engagement -> more desync).
  3. Maintenance theta: log(theta_maintain / theta_baseline_proxy),
     where the proxy is the pre-target window of the same trial.
     Canonical: more positive under load5 than load1.
  4. Load-locked alpha-BOLD: Pearson r across trials between alpha_engaged
     and bold_v1_engaged, split by load.
     Canonical: negative both loads; sub-1070302 predicted to attenuate
     more steeply at load5 (load-dependent collapse).
  5. Vigilance x RT: Pearson r across trials between alpha_engaged and
     reactionTime, pooled across runs.
     Canonical: positive (higher alpha = drowsier = slower).

Output:
  results/vigilance_wm/sub-XXX_run-N_trials.csv   # trial-level rows
  results/vigilance_wm/summary.csv                # per-subject-per-load
  results/vigilance_wm/summary.json
  results/vigilance_wm/by_subject.csv             # pooled-across-runs
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TR = 1.0
HRF_LAG_TR = 5
ALPHA_BAND = (8.0, 12.0)
THETA_BAND = (4.0, 7.0)
HIGHPASS_HZ = 0.01

POSTERIOR_CHS = ["E62", "E67", "E70", "E72", "E75", "E76", "E83", "E84"]
FRONTAL_CHS = ["E11", "E12", "E5", "E6"]

# Trial windows (seconds relative to target onset).
WIN_BASELINE = (-2.0, 0.0)
WIN_ENCODE = (0.0, 2.0)
# engaged + maintenance windows depend on maintDuration per trial.


# ---------------------------------------------------------------------------
# Trial parsing
# ---------------------------------------------------------------------------

def parse_swm_trials(events_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(events_tsv, sep="\t")
    val = df["value"].astype(str)
    targ = df[val.str.startswith("SDRT_targ_l")].copy()
    if targ.empty:
        return targ
    targ["onset_s"] = pd.to_numeric(targ["onsetRelToMRIstart"], errors="coerce")
    targ["load"] = targ["loadName"].map({"load1": 1, "load5": 5})
    targ["accuracy"] = pd.to_numeric(targ.get("accuracy"), errors="coerce")
    rt_col = next((c for c in ("reactionTime", "response_time", "RT")
                   if c in targ.columns), None)
    # reactionTime is already in seconds in this dataset (typical values 0.5-1.5).
    targ["rt"] = (pd.to_numeric(targ[rt_col], errors="coerce")
                  if rt_col else np.nan)
    targ["maint_s"] = pd.to_numeric(targ.get("maintDuration"), errors="coerce")
    keep = ["onset_s", "load", "accuracy", "rt", "maint_s"]
    return targ[keep].dropna(subset=["onset_s", "load"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# EEG envelopes
# ---------------------------------------------------------------------------

def band_log_power(raw, chs: list[str], band: tuple[float, float]) -> tuple[np.ndarray, float]:
    """Return (1-D log-power time-series at raw.sfreq, sfreq)."""
    picks = [c for c in chs if c in raw.ch_names]
    if not picks:
        raise ValueError(f"no requested channels in raw (asked {chs[:4]}...)")
    raw_b = raw.copy().pick(picks).filter(
        l_freq=band[0], h_freq=band[1], verbose=False,
    )
    data = raw_b.get_data()
    env = np.abs(hilbert(data, axis=1))
    log_pow = np.log10(env ** 2 + 1e-12)
    return log_pow.mean(axis=0), raw_b.info["sfreq"]


def window_mean(ts: np.ndarray, sfreq: float, t0_s: float, t1_s: float) -> float:
    """Mean of ts inside [t0_s, t1_s] (seconds, relative to t=0 of ts)."""
    i = max(int(round(t0_s * sfreq)), 0)
    j = min(int(round(t1_s * sfreq)), len(ts))
    if j <= i:
        return float("nan")
    return float(ts[i:j].mean())


# ---------------------------------------------------------------------------
# BOLD V1 time series
# ---------------------------------------------------------------------------

def highpass(ts: np.ndarray) -> np.ndarray:
    fs = 1.0 / TR
    b, a = butter(2, HIGHPASS_HZ / (fs / 2.0), btype="high")
    return filtfilt(b, a, ts)


def build_v1_timeseries(bold_img, v1_path: Path) -> np.ndarray:
    mask_img = compute_epi_mask(bold_img)
    v1 = nib.load(str(v1_path))
    v1_r = resample_to_img(v1, bold_img, interpolation="nearest",
                           force_resample=True, copy_header=True)
    v1_bool = v1_r.get_fdata().astype(bool) & mask_img.get_fdata().astype(bool)
    data = bold_img.get_fdata()
    ts = data[v1_bool, :].mean(axis=0)
    ts = highpass(ts)
    return (ts - ts.mean()) / (ts.std() + 1e-9)


def tr_window_mean(ts: np.ndarray, t0_s: float, t1_s: float) -> float:
    i = max(int(np.floor(t0_s / TR)), 0)
    j = min(int(np.ceil(t1_s / TR)), len(ts))
    if j <= i:
        return float("nan")
    return float(ts[i:j].mean())


# ---------------------------------------------------------------------------
# Per (subject, run)
# ---------------------------------------------------------------------------

def run_one(sub: str, run_id: str, bids_root: Path,
            stimloc_mask_dir: Path, out_dir: Path) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("sub-%s | swm_run-%s", sub, run_id)

    events_tsv = bids_root / f"sub-{sub}" / "ses-02" / "eeg" / \
                 f"sub-{sub}_ses-02_task-swm_run-{run_id}_events.tsv"
    bold_path = bids_root / f"sub-{sub}" / "ses-02" / "func" / \
                f"sub-{sub}_ses-02_task-swm_run-{run_id}.nii.gz"
    v1_path = stimloc_mask_dir / f"sub-{sub}" / \
              f"sub-{sub}_ses-02_stimloc-visual-mask.nii.gz"

    for p, name in [(events_tsv, "events"), (bold_path, "BOLD"),
                    (v1_path, "V1 mask")]:
        if not p.exists():
            log.warning("missing %s: %s", name, p)
            return pd.DataFrame()

    trials = parse_swm_trials(events_tsv)
    log.info("  %d swm trials (load1=%d, load5=%d)",
             len(trials), (trials["load"] == 1).sum(), (trials["load"] == 5).sum())
    if trials.empty:
        return pd.DataFrame()

    raws = get_cleaned_eeg_for_task(bids_root, sub, ses="02", task="swm")
    raw = raws.get(run_id)
    if raw is None:
        log.warning("  no EEG run %s", run_id)
        return pd.DataFrame()

    alpha_ts, eeg_sf = band_log_power(raw, POSTERIOR_CHS, ALPHA_BAND)
    theta_ts, _ = band_log_power(raw, FRONTAL_CHS, THETA_BAND)
    log.info("  EEG envelopes: alpha n=%d, theta n=%d (sfreq=%.1f)",
             len(alpha_ts), len(theta_ts), eeg_sf)

    bold_img, _fd = get_motion_corrected_bold(
        bold_path, sub, ses="02", task="swm", run=run_id,
    )
    v1_ts = build_v1_timeseries(bold_img, v1_path)
    log.info("  V1 BOLD time-series: %d TRs", len(v1_ts))

    rows = []
    for _, t in trials.iterrows():
        onset = float(t["onset_s"])
        maint = float(t["maint_s"]) if np.isfinite(t["maint_s"]) else 4.0
        enc_end = onset + WIN_ENCODE[1]
        engaged_end = onset + WIN_ENCODE[1] + maint
        # EEG windows (in seconds relative to onset).
        a_base = window_mean(alpha_ts, eeg_sf,
                             onset + WIN_BASELINE[0], onset + WIN_BASELINE[1])
        a_enc = window_mean(alpha_ts, eeg_sf,
                            onset + WIN_ENCODE[0], onset + WIN_ENCODE[1])
        a_eng = window_mean(alpha_ts, eeg_sf, onset, engaged_end)
        th_base = window_mean(theta_ts, eeg_sf,
                              onset + WIN_BASELINE[0], onset + WIN_BASELINE[1])
        th_maint = window_mean(theta_ts, eeg_sf, enc_end, engaged_end)
        # BOLD V1 window (HRF-lagged engaged interval).
        bold_eng = tr_window_mean(v1_ts,
                                  onset + HRF_LAG_TR, engaged_end + HRF_LAG_TR)
        rows.append({
            "subject": sub, "run": run_id,
            "onset_s": onset, "load": int(t["load"]),
            "accuracy": float(t["accuracy"]) if np.isfinite(t["accuracy"]) else np.nan,
            "rt_s": float(t["rt"]) if np.isfinite(t["rt"]) else np.nan,
            "maint_s": maint,
            "alpha_baseline": a_base,
            "alpha_encode": a_enc,
            "alpha_engaged": a_eng,
            "theta_baseline": th_base,
            "theta_maintain": th_maint,
            "bold_v1_engaged": bold_eng,
        })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"sub-{sub}_run-{run_id}_trials.csv", index=False)
    log.info("  wrote %d trial rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(all_trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Build per-(sub,load) summary, per-subject pooled summary, and per-subject load contrasts."""
    rows = []
    for (sub, load), g in all_trials.groupby(["subject", "load"]):
        if g.empty:
            continue
        valid_rt = g.loc[g["accuracy"] == 1, "rt_s"]
        d = {
            "subject": sub, "load": int(load), "n_trials": len(g),
            "acc_mean": float(g["accuracy"].mean()),
            "rt_mean_correct_s": float(valid_rt.mean()) if len(valid_rt) else np.nan,
            "rt_sd_correct_s": float(valid_rt.std()) if len(valid_rt) > 1 else np.nan,
            "alpha_desync_log_ratio": float((g["alpha_encode"] - g["alpha_baseline"]).mean()),
            "theta_maint_log_ratio": float((g["theta_maintain"] - g["theta_baseline"]).mean()),
            "mean_alpha_engaged": float(g["alpha_engaged"].mean()),
            "mean_bold_v1_engaged": float(g["bold_v1_engaged"].mean()),
        }
        # Load-locked alpha-BOLD coupling: trial-level Pearson r.
        a, b = g["alpha_engaged"].to_numpy(), g["bold_v1_engaged"].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        d["alpha_bold_r_in_load"] = (
            float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 3
            and a[m].std() > 0 and b[m].std() > 0 else np.nan
        )
        rows.append(d)
    per_load = pd.DataFrame(rows)

    # Per-subject load-contrasts and pooled vigilance correlations.
    contrasts = []
    for sub, g in all_trials.groupby("subject"):
        l1 = g[g["load"] == 1]
        l5 = g[g["load"] == 5]
        l1_correct_rt = l1.loc[l1["accuracy"] == 1, "rt_s"]
        l5_correct_rt = l5.loc[l5["accuracy"] == 1, "rt_s"]
        c = {
            "subject": sub,
            "n_load1": len(l1), "n_load5": len(l5),
            "acc_load1": float(l1["accuracy"].mean()) if len(l1) else np.nan,
            "acc_load5": float(l5["accuracy"].mean()) if len(l5) else np.nan,
            "acc_decrement": float(l1["accuracy"].mean() - l5["accuracy"].mean())
                              if len(l1) and len(l5) else np.nan,
            "rt_load1_s": float(l1_correct_rt.mean()) if len(l1_correct_rt) else np.nan,
            "rt_load5_s": float(l5_correct_rt.mean()) if len(l5_correct_rt) else np.nan,
            "rt_decrement_s": float(l5_correct_rt.mean() - l1_correct_rt.mean())
                              if len(l1_correct_rt) and len(l5_correct_rt) else np.nan,
            # Old (broken) within-trial baseline metrics, kept for diagnostic.
            "alpha_desync_load1": float((l1["alpha_encode"] - l1["alpha_baseline"]).mean())
                                  if len(l1) else np.nan,
            "alpha_desync_load5": float((l5["alpha_encode"] - l5["alpha_baseline"]).mean())
                                  if len(l5) else np.nan,
            "theta_maint_load1": float((l1["theta_maintain"] - l1["theta_baseline"]).mean())
                                 if len(l1) else np.nan,
            "theta_maint_load5": float((l5["theta_maintain"] - l5["theta_baseline"]).mean())
                                 if len(l5) else np.nan,
            # New direct power-comparison metrics: no baseline needed; pre-target
            # baseline was contaminated by back-to-back Sternberg trials.
            "alpha_encode_load1_raw": float(l1["alpha_encode"].mean()) if len(l1) else np.nan,
            "alpha_encode_load5_raw": float(l5["alpha_encode"].mean()) if len(l5) else np.nan,
            "alpha_load_contrast": (
                float(l5["alpha_encode"].mean() - l1["alpha_encode"].mean())
                if len(l1) and len(l5) else np.nan
            ),
            "theta_maintain_load1_raw": float(l1["theta_maintain"].mean()) if len(l1) else np.nan,
            "theta_maintain_load5_raw": float(l5["theta_maintain"].mean()) if len(l5) else np.nan,
            "theta_load_contrast": (
                float(l5["theta_maintain"].mean() - l1["theta_maintain"].mean())
                if len(l1) and len(l5) else np.nan
            ),
        }
        # Pooled-across-runs vigilance x RT (using engaged-window alpha).
        a, r = g["alpha_engaged"].to_numpy(), g["rt_s"].to_numpy()
        m = np.isfinite(a) & np.isfinite(r) & (g["accuracy"].to_numpy() == 1)
        c["vigilance_rt_pearson_r"] = (
            float(np.corrcoef(a[m], r[m])[0, 1]) if m.sum() >= 3
            and a[m].std() > 0 and r[m].std() > 0 else np.nan
        )
        c["vigilance_rt_n"] = int(m.sum())
        contrasts.append(c)
    by_subject = pd.DataFrame(contrasts)

    summary_struct = {
        "per_subject_per_load": rows,
        "by_subject_contrasts": contrasts,
    }
    return per_load, by_subject, summary_struct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_subjects(bids_root: Path) -> list[str]:
    subs = []
    for d in sorted(bids_root.glob("sub-*")):
        if all((d / "ses-02" / "func" /
                f"{d.name}_ses-02_task-swm_run-{r}.nii.gz").exists()
               for r in ("1", "2")):
            subs.append(d.name.replace("sub-", ""))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data", type=Path)
    ap.add_argument("--stimloc-mask-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/vigilance_wm",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--runs", nargs="*", default=["1", "2"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subs = args.subjects or discover_subjects(args.bids_root)
    log.info("Subjects: %s, runs: %s", subs, args.runs)

    all_trials: list[pd.DataFrame] = []
    for s in subs:
        for r in args.runs:
            try:
                df = run_one(s, r, args.bids_root, args.stimloc_mask_dir,
                             args.output_dir)
                if not df.empty:
                    all_trials.append(df)
            except Exception:  # noqa: BLE001
                log.exception("FAILED sub-%s run-%s", s, r)
    if not all_trials:
        log.error("no trials extracted; aborting")
        return
    trials_df = pd.concat(all_trials, ignore_index=True)
    trials_df.to_csv(args.output_dir / "all_trials.csv", index=False)
    log.info("Total trials extracted: %d", len(trials_df))

    per_load, by_subject, summary_struct = summarise(trials_df)
    per_load.to_csv(args.output_dir / "summary.csv", index=False)
    by_subject.to_csv(args.output_dir / "by_subject.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_struct, indent=2)
    )

    # ------------------------------------------------------------------
    # Pretty-print the five sub-analyses
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("ANALYSIS 1 — Behaviour (accuracy and RT on correct trials)")
    print("=" * 92)
    print(f"{'Subject':<10} {'n L1':>5} {'n L5':>5} "
          f"{'acc L1':>8} {'acc L5':>8} {'Δ acc':>7} "
          f"{'RT L1 (s)':>10} {'RT L5 (s)':>10} {'Δ RT (s)':>10}")
    for _, r in by_subject.iterrows():
        print(f"{r['subject']:<10} {r['n_load1']:>5} {r['n_load5']:>5} "
              f"{r['acc_load1']:>8.3f} {r['acc_load5']:>8.3f} "
              f"{r['acc_decrement']:>+7.3f} "
              f"{r['rt_load1_s']:>10.3f} {r['rt_load5_s']:>10.3f} "
              f"{r['rt_decrement_s']:>+10.3f}")

    print("\n" + "=" * 92)
    print("ANALYSIS 2 — Alpha desynchronisation (DIRECT load5 vs load1, no baseline)")
    print("Canonical: alpha encode_L5 < encode_L1 (more desync under high WM load).")
    print("Δ should be NEGATIVE.")
    print("=" * 92)
    print(f"{'Subject':<10} {'α enc L1 (raw)':>15} {'α enc L5 (raw)':>15} {'Δ (L5 - L1)':>14}")
    for _, r in by_subject.iterrows():
        print(f"{r['subject']:<10} {r['alpha_encode_load1_raw']:>+15.4f} "
              f"{r['alpha_encode_load5_raw']:>+15.4f} "
              f"{r['alpha_load_contrast']:>+14.4f}")

    print("\n" + "=" * 92)
    print("ANALYSIS 3 — Frontal-midline theta maintenance (DIRECT load5 vs load1)")
    print("Canonical: theta_L5 > theta_L1 (more sustained theta under high load, Jensen 2002).")
    print("Δ should be POSITIVE.")
    print("=" * 92)
    print(f"{'Subject':<10} {'θ maint L1 (raw)':>17} {'θ maint L5 (raw)':>17} {'Δ (L5 - L1)':>14}")
    for _, r in by_subject.iterrows():
        print(f"{r['subject']:<10} {r['theta_maintain_load1_raw']:>+17.4f} "
              f"{r['theta_maintain_load5_raw']:>+17.4f} "
              f"{r['theta_load_contrast']:>+14.4f}")

    print("\n" + "=" * 92)
    print("ANALYSIS 2b / 3b — diagnostic only: old broken metrics (contaminated baseline)")
    print("=" * 92)
    print(f"{'Subject':<10} {'α desync L1':>12} {'α desync L5':>12} "
          f"{'θ maint L1':>12} {'θ maint L5':>12}")
    for _, r in by_subject.iterrows():
        print(f"{r['subject']:<10} {r['alpha_desync_load1']:>+12.4f} "
              f"{r['alpha_desync_load5']:>+12.4f} "
              f"{r['theta_maint_load1']:>+12.4f} {r['theta_maint_load5']:>+12.4f}")

    print("\n" + "=" * 92)
    print("ANALYSIS 4 — Trial-level alpha-BOLD coupling, split by load")
    print("Canonical: negative (more alpha -> less V1 BOLD across trials).")
    print("=" * 92)
    print(f"{'Subject':<10} {'load':>5} {'n trials':>9} {'r(alpha, V1 BOLD)':>20}")
    for _, r in per_load.iterrows():
        print(f"{r['subject']:<10} {r['load']:>5} {r['n_trials']:>9} "
              f"{r['alpha_bold_r_in_load']:>+20.3f}")

    print("\n" + "=" * 92)
    print("ANALYSIS 5 — Vigilance x RT (alpha_engaged vs reactionTime, correct trials, pooled)")
    print("Canonical: positive r (higher alpha = drowsier = slower).")
    print("=" * 92)
    print(f"{'Subject':<10} {'n trials':>9} {'r(alpha, RT)':>14}")
    for _, r in by_subject.iterrows():
        print(f"{r['subject']:<10} {r['vigilance_rt_n']:>9} "
              f"{r['vigilance_rt_pearson_r']:>+14.3f}")

    print("\n" + "=" * 92)
    print("Headline prediction: vigilance-vulnerable subject (sub-1070302) shows")
    print("  (i)  larger Δ accuracy and Δ RT under load5,")
    print("  (ii) attenuated alpha desync at load5 (failure to disengage default state),")
    print("  (iii) attenuated theta maintenance increase at load5,")
    print("  (iv) load-dependent alpha-BOLD collapse (canonical at load1, near-zero at load5),")
    print("  (v)  weaker vigilance x RT correlation (alpha doesn't predict performance).")


if __name__ == "__main__":
    main()
