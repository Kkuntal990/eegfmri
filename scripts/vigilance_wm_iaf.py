"""IAF (Individual Alpha Frequency) sensitivity test for Analysis 4.

Addresses Risk C of reviewer feedback: a fixed 8-12 Hz alpha filter may
distort the alpha envelope if a subject's alpha peak shifts under WM
load (Klimesch IAF tradition; Liu et al. Sternberg).

Pipeline:

  1. Compute per-subject IAF from the rest-task cleaned EEG by finding the
     posterior PSD peak in 7-13 Hz (Welch's method).
  2. Re-extract the trial-locked alpha envelope on swm_run-1 / swm_run-2
     using a subject-specific [IAF - 2, IAF + 2] Hz filter instead of
     the fixed 8-12 Hz band.
  3. Re-run the Bridwell lag sweep (lags 3, 5, 7, 9, 11 TR) on the trial-
     level Pearson r between IAF-band alpha and V1-mean BOLD, split by load.
  4. Output side-by-side the fixed-band and IAF-band lag matrices so we
     can read directly whether the headline survives the IAF correction.

If the load x lag x phenotype pattern reproduces with IAF filtering,
Risk C is closed methodologically. If it changes substantially, the
proposal must be updated.

Output:
  results/vigilance_wm_iaf/iaf_per_subject.json     # IAF per subject
  results/vigilance_wm_iaf/iaf_trials.csv           # trial-level IAF-band features
  results/vigilance_wm_iaf/iaf_lag_sweep.csv        # the IAF-band lag matrix
  results/vigilance_wm_iaf/comparison.csv           # side-by-side vs fixed-band

Usage:
    python scripts/vigilance_wm_iaf.py \
        --bids-root /work/hdd/bbnv/kuntal/eegfmri_data \
        --stimloc-mask-dir /projects/bbnv/kkokate/eegfmri/results/stimloc_mask \
        --fixed-lag-sweep /projects/bbnv/kkokate/eegfmri/results/vigilance_wm/lag_sweep.csv \
        --output-dir /projects/bbnv/kkokate/eegfmri/results/vigilance_wm_iaf
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
from scipy.signal import butter, filtfilt, hilbert, welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cache import get_cleaned_eeg_for_task, get_motion_corrected_bold  # noqa: E402
from vigilance_wm_trials import (  # noqa: E402
    HRF_LAGS_TO_TEST, POSTERIOR_CHS, TR, HIGHPASS_HZ,
    parse_swm_trials, window_mean, tr_window_mean, highpass,
    build_v1_timeseries,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IAF_SEARCH_BAND = (7.0, 13.0)
IAF_HALF_BAND = 2.0          # filter at IAF +/- 2 Hz
SMOOTH_WIN_S = 5.0
WIN_BASELINE = (-2.0, 0.0)
WIN_ENCODE = (0.0, 2.0)


def compute_iaf(raw, chs: list[str]) -> dict:
    """Find Individual Alpha Frequency after removing 1/f aperiodic component.

    Per Corcoran et al. 2018 / FOOOF tradition: a raw PSD peak in 7-13 Hz
    is dominated by the 1/f slope (highest power is at lowest frequency),
    so we first fit log(PSD) ~ log(freq) on the aperiodic background
    (2-50 Hz, excluding 7-13 Hz), subtract it, then find the peak in the
    periodic residual. This addresses the FOOOF / aperiodic-1/f confound
    (Donoghue 2020, Waschke 2021).
    """
    picks = [c for c in chs if c in raw.ch_names]
    if not picks:
        raise ValueError("no posterior channels in raw")
    raw_p = raw.copy().pick(picks)
    data = raw_p.get_data()
    sfreq = raw_p.info["sfreq"]
    freqs, psd = welch(data, fs=sfreq, nperseg=int(sfreq * 4),
                       noverlap=int(sfreq * 2), axis=1)
    psd_mean = psd.mean(axis=0)

    valid = (freqs > 1) & (freqs < 50)
    f_v = freqs[valid]
    p_v = psd_mean[valid]
    log_f = np.log10(f_v)
    log_p = np.log10(p_v + 1e-20)
    alpha_mask = (f_v >= IAF_SEARCH_BAND[0]) & (f_v <= IAF_SEARCH_BAND[1])
    fit_mask = ~alpha_mask  # fit 1/f on everything except the alpha band
    slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
    aperiodic_log = slope * log_f + intercept
    # Periodic residual in linear power units; clip negatives at 0.
    periodic = np.clip(p_v - 10 ** aperiodic_log, 0.0, None)

    band_freqs = f_v[alpha_mask]
    band_periodic = periodic[alpha_mask]
    band_raw_power = p_v[alpha_mask]
    if not len(band_periodic) or band_periodic.sum() == 0:
        return {"iaf_hz": float("nan"), "channels_used": picks}
    peak_idx = int(np.argmax(band_periodic))
    iaf = float(band_freqs[peak_idx])
    com = float((band_freqs * band_periodic).sum() / band_periodic.sum())
    # Also return the raw-PSD peak for diagnostic (this was the buggy
    # value before the 1/f correction was added).
    raw_peak_idx = int(np.argmax(band_raw_power))
    raw_peak_iaf = float(band_freqs[raw_peak_idx])
    return {
        "iaf_hz": iaf,                         # 1/f-corrected peak (primary)
        "iaf_com_hz": com,                     # 1/f-corrected centre-of-mass
        "iaf_raw_peak_hz": raw_peak_iaf,       # diagnostic: pre-correction peak
        "peak_periodic_power": float(band_periodic[peak_idx]),
        "aperiodic_slope": float(slope),
        "channels_used": picks,
    }


def iaf_log_power(raw, chs: list[str], iaf: float,
                  half_band: float = IAF_HALF_BAND) -> tuple[np.ndarray, float]:
    """Compute IAF-band log power envelope at raw.sfreq."""
    picks = [c for c in chs if c in raw.ch_names]
    raw_b = raw.copy().pick(picks).filter(
        l_freq=max(iaf - half_band, 4.0),
        h_freq=iaf + half_band,
        verbose=False,
    )
    data = raw_b.get_data()
    env = np.abs(hilbert(data, axis=1))
    log_pow = np.log10(env ** 2 + 1e-12)
    return log_pow.mean(axis=0), raw_b.info["sfreq"]


def run_one(sub: str, run_id: str, bids_root: Path,
            stimloc_mask_dir: Path, iaf: float, out_dir: Path) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("sub-%s | swm_run-%s | IAF = %.2f Hz", sub, run_id, iaf)
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
    if trials.empty:
        return pd.DataFrame()

    raws = get_cleaned_eeg_for_task(bids_root, sub, ses="02", task="swm")
    raw = raws.get(run_id)
    if raw is None:
        return pd.DataFrame()

    alpha_ts, eeg_sf = iaf_log_power(raw, POSTERIOR_CHS, iaf)
    log.info("  IAF-band envelope: %.1f-%.1f Hz, n=%d (sfreq=%.1f)",
             max(iaf - IAF_HALF_BAND, 4.0), iaf + IAF_HALF_BAND,
             len(alpha_ts), eeg_sf)

    bold_img, _fd = get_motion_corrected_bold(
        bold_path, sub, ses="02", task="swm", run=run_id,
    )
    v1_ts = build_v1_timeseries(bold_img, v1_path)

    rows = []
    for _, t in trials.iterrows():
        onset = float(t["onset_s"])
        maint = float(t["maint_s"]) if np.isfinite(t["maint_s"]) else 4.0
        engaged_end = onset + WIN_ENCODE[1] + maint
        a_eng = window_mean(alpha_ts, eeg_sf, onset, engaged_end)
        bold_by_lag = {
            lag: tr_window_mean(v1_ts, onset + lag, engaged_end + lag)
            for lag in HRF_LAGS_TO_TEST
        }
        row = {
            "subject": sub, "run": run_id,
            "onset_s": onset, "load": int(t["load"]),
            "accuracy": float(t["accuracy"]) if np.isfinite(t["accuracy"]) else np.nan,
            "rt_s": float(t["rt"]) if np.isfinite(t["rt"]) else np.nan,
            "iaf_hz": iaf,
            "alpha_iaf_engaged": a_eng,
        }
        for lag, v in bold_by_lag.items():
            row[f"bold_v1_engaged_lag{lag}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"sub-{sub}_run-{run_id}_iaf_trials.csv", index=False)
    return df


def lag_sweep(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sub, load), g in trials.groupby(["subject", "load"]):
        a = g["alpha_iaf_engaged"].to_numpy()
        d = {"subject": sub, "load": int(load), "n_trials": len(g),
             "iaf_hz": float(g["iaf_hz"].iloc[0])}
        for lag in HRF_LAGS_TO_TEST:
            b = g[f"bold_v1_engaged_lag{lag}"].to_numpy()
            m = np.isfinite(a) & np.isfinite(b)
            d[f"r_lag{lag}"] = (
                float(np.corrcoef(a[m], b[m])[0, 1])
                if m.sum() >= 3 and a[m].std() > 0 and b[m].std() > 0
                else float("nan")
            )
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root",
                    default="/work/hdd/bbnv/kuntal/eegfmri_data", type=Path)
    ap.add_argument("--stimloc-mask-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/stimloc_mask",
                    type=Path)
    ap.add_argument("--output-dir",
                    default="/projects/bbnv/kkokate/eegfmri/results/vigilance_wm_iaf",
                    type=Path)
    ap.add_argument("--fixed-lag-sweep",
                    default="/projects/bbnv/kkokate/eegfmri/results/vigilance_wm/lag_sweep.csv",
                    type=Path)
    ap.add_argument("--subjects", nargs="*", default=["500", "1070302"])
    ap.add_argument("--runs", nargs="*", default=["1", "2"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: compute per-subject IAF from the REST task (clean baseline).
    iaf_per_sub: dict[str, dict] = {}
    for sub in args.subjects:
        log.info("Computing IAF for sub-%s from rest task", sub)
        raws = get_cleaned_eeg_for_task(args.bids_root, sub, ses="02", task="rest")
        if not raws:
            log.warning("  no rest EEG; skipping IAF computation for sub-%s", sub)
            continue
        raw = next(iter(raws.values()))
        info = compute_iaf(raw, POSTERIOR_CHS)
        iaf_per_sub[sub] = info
        log.info("  sub-%s IAF = %.2f Hz (centre-of-mass = %.2f Hz, "
                 "periodic peak power = %.2g, raw-peak diag = %.2f Hz, 1/f slope = %+.3f)",
                 sub, info["iaf_hz"], info["iaf_com_hz"],
                 info.get("peak_periodic_power", float("nan")),
                 info.get("iaf_raw_peak_hz", float("nan")),
                 info.get("aperiodic_slope", float("nan")))
    (args.output_dir / "iaf_per_subject.json").write_text(
        json.dumps(iaf_per_sub, indent=2)
    )

    # Step 2: re-extract trial-level features with IAF-band filter.
    all_trials = []
    for sub in args.subjects:
        if sub not in iaf_per_sub:
            continue
        iaf = iaf_per_sub[sub]["iaf_hz"]
        for r in args.runs:
            try:
                df = run_one(sub, r, args.bids_root, args.stimloc_mask_dir,
                             iaf, args.output_dir)
                if not df.empty:
                    all_trials.append(df)
            except Exception:  # noqa: BLE001
                log.exception("FAILED sub-%s run-%s", sub, r)
    if not all_trials:
        log.error("no trials extracted; aborting")
        return
    trials_df = pd.concat(all_trials, ignore_index=True)
    trials_df.to_csv(args.output_dir / "iaf_trials.csv", index=False)

    # Step 3: lag sweep with IAF-band.
    iaf_lag_df = lag_sweep(trials_df)
    iaf_lag_df.to_csv(args.output_dir / "iaf_lag_sweep.csv", index=False)

    # Step 4: side-by-side comparison with fixed-band lag sweep.
    if args.fixed_lag_sweep.exists():
        fixed_df = pd.read_csv(args.fixed_lag_sweep)
        # Long-format for clean comparison.
        merged_rows = []
        for _, ir in iaf_lag_df.iterrows():
            fixed_row = fixed_df[
                (fixed_df["subject"].astype(str) == str(ir["subject"]))
                & (fixed_df["load"] == ir["load"])
            ]
            if fixed_row.empty:
                continue
            fr = fixed_row.iloc[0]
            for lag in HRF_LAGS_TO_TEST:
                merged_rows.append({
                    "subject": ir["subject"], "load": int(ir["load"]),
                    "lag_tr": lag,
                    "iaf_hz": ir["iaf_hz"],
                    "r_fixed_band": float(fr[f"r_lag{lag}"]),
                    "r_iaf_band": float(ir[f"r_lag{lag}"]),
                    "delta_r": float(ir[f"r_lag{lag}"]) - float(fr[f"r_lag{lag}"]),
                    "sign_changed": np.sign(ir[f"r_lag{lag}"]) != np.sign(fr[f"r_lag{lag}"]),
                })
        comp_df = pd.DataFrame(merged_rows)
        comp_df.to_csv(args.output_dir / "comparison.csv", index=False)

    # ------------------------------------------------------------------
    # Print: IAFs, IAF-band lag matrix, side-by-side comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("INDIVIDUAL ALPHA FREQUENCY (1/f-corrected peak in 7-13 Hz over posterior channels)")
    print("=" * 92)
    print(f"{'Subject':<10} {'IAF (Hz)':>10} {'IAF-CoM (Hz)':>14} "
          f"{'raw-peak diag':>14} {'1/f slope':>11} {'Band used (Hz)':>18}")
    for sub, info in iaf_per_sub.items():
        band = (max(info["iaf_hz"] - IAF_HALF_BAND, 4.0),
                info["iaf_hz"] + IAF_HALF_BAND)
        print(f"{sub:<10} {info['iaf_hz']:>10.2f} {info['iaf_com_hz']:>14.2f} "
              f"{info.get('iaf_raw_peak_hz', float('nan')):>14.2f} "
              f"{info.get('aperiodic_slope', float('nan')):>+11.3f} "
              f"{band[0]:>5.1f} - {band[1]:.1f}")

    print("\n" + "=" * 96)
    print("LAG SWEEP — IAF-BAND alpha-BOLD coupling at lags 3-11 TR")
    print("=" * 96)
    print(f"{'Subject':<10} {'load':>5} {'IAF':>6} "
          + " ".join(f"r_lag{lag:>2}" for lag in HRF_LAGS_TO_TEST))
    for _, r in iaf_lag_df.iterrows():
        vals = " ".join(f"{r[f'r_lag{lag}']:>+7.3f}" for lag in HRF_LAGS_TO_TEST)
        print(f"{str(r['subject']):<10} {int(r['load']):>5} "
              f"{r['iaf_hz']:>6.2f}  {vals}")

    if args.fixed_lag_sweep.exists():
        print("\n" + "=" * 96)
        print("SIDE-BY-SIDE: fixed 8-12 Hz vs IAF +/- 2 Hz lag matrices")
        print("Δr = IAF - fixed; SIGN★ = sign changed between bands.")
        print("=" * 96)
        print(f"{'Subject':<10} {'load':>5} {'lag':>4} "
              f"{'r_fixed':>9} {'r_IAF':>8} {'Δr':>7} {'sign★':>7}")
        for _, r in comp_df.iterrows():
            star = "YES" if r["sign_changed"] else ""
            print(f"{str(r['subject']):<10} {int(r['load']):>5} "
                  f"{int(r['lag_tr']):>4} "
                  f"{r['r_fixed_band']:>+9.3f} {r['r_iaf_band']:>+8.3f} "
                  f"{r['delta_r']:>+7.3f} {star:>7}")

    print("\n" + "=" * 96)
    print("VERDICT:")
    print("  If the load x lag pattern for sub-1070302 (positive at lag 5, negative")
    print("  at lag 11 for load1; positive everywhere for load5) reproduces with the")
    print("  IAF-band filter, Risk C is closed and the proposal headline holds.")
    print("  If the signs flip in many cells, the fixed-band filter was distorting")
    print("  the effect and the proposal must be updated with IAF-only conclusions.")


if __name__ == "__main__":
    main()
