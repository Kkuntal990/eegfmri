"""Validate EEG data quality using alpha power analysis and simple classifiers.

Literature-backed validation (Barry et al., 2007; Grandy et al., 2013):
- EC (eyes closed) should show significantly higher alpha (8-13 Hz) power
  than EO (eyes open) at posterior/occipital sites (Berger's effect).
- Expected: Cohen's d > 1.0, p < 0.001 at occipital channels.
- CSP+LDA within-subject accuracy should be >85% for meaningful data.
- SVM on band-power features should also exceed 80%.

If alpha power does NOT differ, the data is likely corrupted or not meaningful.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import mne
import numpy as np
from scipy import stats
from scipy.signal import welch

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/projects/bbnv/kkokate/eegfmri")

from eegfmri_loader import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_MAP = {"EC": 0, "EO": 1}

# EGI HydroCel 128 posterior channels (occipital + parietal)
# Approximate 10-20 equivalents:
#   Occipital: E70≈O1, E75≈Oz, E83≈O2
#   Parieto-occipital: E71, E72, E76, E77, E66, E84
#   Parietal: E52≈P3, E62≈Pz, E92≈P4
OCCIPITAL_CHS = ["E70", "E71", "E74", "E75", "E76", "E82", "E83"]
PARIETAL_CHS = ["E52", "E60", "E61", "E62", "E67", "E72", "E77", "E78", "E85", "E86"]
POSTERIOR_CHS = OCCIPITAL_CHS + PARIETAL_CHS
FRONTAL_CHS = ["E11", "E12", "E19", "E20", "E23", "E24", "E27", "E28"]  # control region


FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def extract_epochs(dataset, window_size_samples=500):
    """Extract epochs grouped by subject and session."""
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    tmax = (window_size_samples - 1) / sfreq

    results = []  # list of dicts with subject, session, X, y

    for ds in dataset.datasets:
        raw = ds.raw
        subject = ds.description["subject"]
        session = ds.description["session"]

        events, event_id = mne.events_from_annotations(raw)
        keep_ids = {k: v for k, v in event_id.items() if k in TARGET_MAP}
        if not keep_ids:
            logger.warning("No EC/EO events for sub-%s ses-%s, skipping", subject, session)
            continue

        epochs = mne.Epochs(
            raw, events, keep_ids,
            tmin=0, tmax=tmax, baseline=None,
            preload=True, reject=None, verbose=False,
            event_repeated="drop",
        )

        X = epochs.get_data(copy=True).astype(np.float32)
        inv_event_id = {v: k for k, v in keep_ids.items()}
        y = np.array([TARGET_MAP[inv_event_id[eid]] for eid in epochs.events[:, 2]])

        logger.info("sub-%s ses-%s: %d epochs (EC=%d, EO=%d)",
                     subject, session, len(y), (y == 0).sum(), (y == 1).sum())

        results.append({
            "subject": subject,
            "session": session,
            "X": X,
            "y": y,
            "ch_names": epochs.ch_names,
            "sfreq": sfreq,
        })

    return results


def compute_band_power(X, sfreq, fmin, fmax):
    """Compute mean band power per epoch per channel using Welch PSD.

    Args:
        X: (n_epochs, n_channels, n_times)
        sfreq: sampling frequency
        fmin, fmax: band limits

    Returns:
        (n_epochs, n_channels) array of mean power in band
    """
    n_epochs, n_channels, n_times = X.shape
    nperseg = min(256, n_times)

    powers = np.zeros((n_epochs, n_channels))
    for i in range(n_epochs):
        freqs, psd = welch(X[i], fs=sfreq, nperseg=nperseg)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        powers[i] = np.mean(psd[:, freq_mask], axis=-1)

    return powers


def get_channel_indices(ch_names, target_chs):
    """Get indices of target channels that exist in ch_names."""
    indices = []
    found = []
    for ch in target_chs:
        if ch in ch_names:
            indices.append(ch_names.index(ch))
            found.append(ch)
    return indices, found


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.std(group1, ddof=1)**2 +
                           (n2 - 1) * np.std(group2, ddof=1)**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def alpha_power_analysis(data, output_dir):
    """Run alpha power statistical analysis for each subject/session.

    Tests whether EC alpha power > EO alpha power at posterior channels.
    """
    logger.info("=" * 70)
    logger.info("ALPHA POWER ANALYSIS")
    logger.info("=" * 70)

    all_stats = []

    for rec in data:
        subj = rec["subject"]
        sess = rec["session"]
        X, y = rec["X"], rec["y"]
        ch_names = list(rec["ch_names"])
        sfreq = rec["sfreq"]

        logger.info("\n--- sub-%s ses-%s ---", subj, sess)

        # Compute alpha power
        alpha_power = compute_band_power(X, sfreq, 8, 13)  # (n_epochs, n_channels)

        # Log-transform for normality
        alpha_log = np.log10(alpha_power + 1e-20)

        ec_mask = y == 0
        eo_mask = y == 1

        if ec_mask.sum() == 0 or eo_mask.sum() == 0:
            logger.warning("  Missing EC or EO epochs, skipping stats")
            continue

        ec_alpha = alpha_log[ec_mask]
        eo_alpha = alpha_log[eo_mask]

        # Get channel indices for regions of interest
        occ_idx, occ_found = get_channel_indices(ch_names, OCCIPITAL_CHS)
        par_idx, par_found = get_channel_indices(ch_names, PARIETAL_CHS)
        post_idx, post_found = get_channel_indices(ch_names, POSTERIOR_CHS)
        front_idx, front_found = get_channel_indices(ch_names, FRONTAL_CHS)

        rec_stats = {
            "subject": subj,
            "session": sess,
            "n_ec": int(ec_mask.sum()),
            "n_eo": int(eo_mask.sum()),
            "regions": {},
        }

        # Analyze each region
        for region_name, idx_list, found_list in [
            ("occipital", occ_idx, occ_found),
            ("parietal", par_idx, par_found),
            ("posterior", post_idx, post_found),
            ("frontal", front_idx, front_found),
            ("all_channels", list(range(len(ch_names))), ch_names),
        ]:
            if not idx_list:
                logger.warning("  No %s channels found", region_name)
                continue

            # Mean across channels in region
            ec_region = np.mean(ec_alpha[:, idx_list], axis=1)
            eo_region = np.mean(eo_alpha[:, idx_list], axis=1)

            t_stat, p_val = stats.ttest_ind(ec_region, eo_region, alternative="greater")
            d = cohens_d(ec_region, eo_region)

            # Raw power ratio (not log)
            ec_raw = np.mean(alpha_power[ec_mask][:, idx_list])
            eo_raw = np.mean(alpha_power[eo_mask][:, idx_list])
            ratio = ec_raw / eo_raw if eo_raw > 0 else float("inf")

            region_stats = {
                "n_channels": len(idx_list),
                "ec_mean_log_power": float(np.mean(ec_region)),
                "eo_mean_log_power": float(np.mean(eo_region)),
                "ec_raw_power": float(ec_raw),
                "eo_raw_power": float(eo_raw),
                "ec_eo_ratio": float(ratio),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
            }
            rec_stats["regions"][region_name] = region_stats

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            logger.info("  %s (%d ch): EC/EO ratio=%.2f, d=%.2f, t=%.2f, p=%.1e %s",
                        region_name, len(idx_list), ratio, d, t_stat, p_val, sig)

        # Per-channel analysis: find top channels with largest EC-EO difference
        ec_mean = np.mean(ec_alpha, axis=0)
        eo_mean = np.mean(eo_alpha, axis=0)
        diff = ec_mean - eo_mean  # positive = EC > EO (expected for alpha)
        top_idx = np.argsort(diff)[::-1][:10]

        logger.info("  Top 10 channels (EC > EO alpha):")
        top_channels = []
        for rank, ci in enumerate(top_idx):
            ch_d = cohens_d(ec_alpha[:, ci], eo_alpha[:, ci])
            logger.info("    %d. %s: diff=%.3f, d=%.2f", rank + 1, ch_names[ci], diff[ci], ch_d)
            top_channels.append({"channel": ch_names[ci], "diff": float(diff[ci]), "cohens_d": float(ch_d)})
        rec_stats["top_channels_ec_gt_eo"] = top_channels

        all_stats.append(rec_stats)

    return all_stats


def band_power_features(X, sfreq):
    """Extract band power features for all frequency bands.

    Returns (n_epochs, n_channels * n_bands) feature matrix.
    """
    features = []
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        bp = compute_band_power(X, sfreq, fmin, fmax)
        features.append(np.log10(bp + 1e-20))
    return np.hstack(features)


def classification_analysis(data, output_dir):
    """Run within-subject classification using CSP+LDA and SVM.

    Uses 5-fold stratified cross-validation.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CLASSIFICATION ANALYSIS (within-subject)")
    logger.info("=" * 70)

    all_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for rec in data:
        subj = rec["subject"]
        sess = rec["session"]
        X, y = rec["X"], rec["y"]
        sfreq = rec["sfreq"]

        logger.info("\n--- sub-%s ses-%s (%d epochs: EC=%d, EO=%d) ---",
                     subj, sess, len(y), (y == 0).sum(), (y == 1).sum())

        if (y == 0).sum() < 5 or (y == 1).sum() < 5:
            logger.warning("  Too few epochs for CV, skipping")
            continue

        n_splits = min(5, min((y == 0).sum(), (y == 1).sum()))
        if n_splits < 5:
            cv_local = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv_local = cv

        rec_results = {
            "subject": subj,
            "session": sess,
            "n_epochs": len(y),
            "n_ec": int((y == 0).sum()),
            "n_eo": int((y == 1).sum()),
            "classifiers": {},
        }

        # 1. CSP + LDA
        try:
            n_components = min(6, X.shape[1] - 1)
            csp_lda = Pipeline([
                ("csp", mne.decoding.CSP(n_components=n_components, reg=None, log=True)),
                ("lda", LinearDiscriminantAnalysis()),
            ])
            scores_csp = cross_val_score(csp_lda, X, y, cv=cv_local, scoring="accuracy")
            bal_scores_csp = cross_val_score(csp_lda, X, y, cv=cv_local, scoring="balanced_accuracy")
            rec_results["classifiers"]["CSP_LDA"] = {
                "accuracy_mean": float(np.mean(scores_csp)),
                "accuracy_std": float(np.std(scores_csp)),
                "balanced_accuracy_mean": float(np.mean(bal_scores_csp)),
                "balanced_accuracy_std": float(np.std(bal_scores_csp)),
                "fold_accuracies": scores_csp.tolist(),
            }
            logger.info("  CSP+LDA: acc=%.1f%% (+/-%.1f%%), bal_acc=%.1f%%",
                        np.mean(scores_csp) * 100, np.std(scores_csp) * 100,
                        np.mean(bal_scores_csp) * 100)
        except Exception as e:
            logger.error("  CSP+LDA failed: %s", e)
            rec_results["classifiers"]["CSP_LDA"] = {"error": str(e)}

        # 2. SVM on band power features
        try:
            feats = band_power_features(X, sfreq)
            svm_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0)),
            ])
            scores_svm = cross_val_score(svm_pipe, feats, y, cv=cv_local, scoring="accuracy")
            bal_scores_svm = cross_val_score(svm_pipe, feats, y, cv=cv_local, scoring="balanced_accuracy")
            rec_results["classifiers"]["SVM_bandpower"] = {
                "accuracy_mean": float(np.mean(scores_svm)),
                "accuracy_std": float(np.std(scores_svm)),
                "balanced_accuracy_mean": float(np.mean(bal_scores_svm)),
                "balanced_accuracy_std": float(np.std(bal_scores_svm)),
                "fold_accuracies": scores_svm.tolist(),
            }
            logger.info("  SVM (band power): acc=%.1f%% (+/-%.1f%%), bal_acc=%.1f%%",
                        np.mean(scores_svm) * 100, np.std(scores_svm) * 100,
                        np.mean(bal_scores_svm) * 100)
        except Exception as e:
            logger.error("  SVM band power failed: %s", e)
            rec_results["classifiers"]["SVM_bandpower"] = {"error": str(e)}

        # 3. LDA on alpha power only (simplest possible)
        try:
            alpha_feats = compute_band_power(X, sfreq, 8, 13)
            alpha_feats = np.log10(alpha_feats + 1e-20)
            lda_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lda", LinearDiscriminantAnalysis()),
            ])
            scores_lda = cross_val_score(lda_pipe, alpha_feats, y, cv=cv_local, scoring="accuracy")
            bal_scores_lda = cross_val_score(lda_pipe, alpha_feats, y, cv=cv_local, scoring="balanced_accuracy")
            rec_results["classifiers"]["LDA_alpha_only"] = {
                "accuracy_mean": float(np.mean(scores_lda)),
                "accuracy_std": float(np.std(scores_lda)),
                "balanced_accuracy_mean": float(np.mean(bal_scores_lda)),
                "balanced_accuracy_std": float(np.std(bal_scores_lda)),
                "fold_accuracies": scores_lda.tolist(),
            }
            logger.info("  LDA (alpha only): acc=%.1f%% (+/-%.1f%%), bal_acc=%.1f%%",
                        np.mean(scores_lda) * 100, np.std(scores_lda) * 100,
                        np.mean(bal_scores_lda) * 100)
        except Exception as e:
            logger.error("  LDA alpha failed: %s", e)
            rec_results["classifiers"]["LDA_alpha_only"] = {"error": str(e)}

        all_results.append(rec_results)

    return all_results


def psd_summary(data, output_dir):
    """Compute and log PSD summary per condition for each recording."""
    logger.info("\n" + "=" * 70)
    logger.info("PSD SUMMARY PER CONDITION")
    logger.info("=" * 70)

    all_psd = []

    for rec in data:
        subj = rec["subject"]
        sess = rec["session"]
        X, y = rec["X"], rec["y"]
        sfreq = rec["sfreq"]

        ec_mask = y == 0
        eo_mask = y == 1

        if ec_mask.sum() == 0 or eo_mask.sum() == 0:
            continue

        rec_psd = {"subject": subj, "session": sess, "bands": {}}

        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            bp = compute_band_power(X, sfreq, fmin, fmax)  # (n_epochs, n_ch)
            ec_mean = float(np.mean(bp[ec_mask]))
            eo_mean = float(np.mean(bp[eo_mask]))
            ratio = ec_mean / eo_mean if eo_mean > 0 else float("inf")

            rec_psd["bands"][band_name] = {
                "ec_mean": ec_mean,
                "eo_mean": eo_mean,
                "ec_eo_ratio": ratio,
            }

            logger.info("  sub-%s ses-%s %s: EC=%.2e, EO=%.2e, ratio=%.2f",
                        subj, sess, band_name, ec_mean, eo_mean, ratio)

        all_psd.append(rec_psd)

    return all_psd


def main():
    parser = argparse.ArgumentParser(description="Validate EEG data quality")
    parser.add_argument("--bids-root", default="/work/hdd/bbnv/kuntal/eegfmri_data")
    parser.add_argument("--output-dir", default="/projects/bbnv/kkokate/eegfmri/results/validation")
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--window-size", type=int, default=500,
                        help="Window size in samples (default: 500 = 2s at 250Hz)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading eoec data from %s", args.bids_root)
    dataset = load_dataset(
        args.bids_root,
        tasks=["eoec"],
        sessions=args.sessions,
        preprocess=True,
        resample_freq=250.0,
        pick_eeg_only=True,
    )
    logger.info("Loaded %d recordings", len(dataset.datasets))

    # Drop flat Cz reference
    for ds in dataset.datasets:
        if "Cz" in ds.raw.ch_names:
            ds.raw.drop_channels(["Cz"])

    # ---- Extract epochs per recording ----
    data = extract_epochs(dataset, window_size_samples=args.window_size)

    if not data:
        logger.error("No valid recordings found!")
        sys.exit(1)

    # ---- Analysis 1: Alpha power statistics ----
    alpha_stats = alpha_power_analysis(data, output_dir)

    # ---- Analysis 2: PSD summary across bands ----
    psd_stats = psd_summary(data, output_dir)

    # ---- Analysis 3: Classification ----
    clf_results = classification_analysis(data, output_dir)

    # ---- Final verdict ----
    logger.info("\n" + "=" * 70)
    logger.info("DATA QUALITY VERDICT")
    logger.info("=" * 70)

    meaningful_count = 0
    total_count = 0

    for stat in alpha_stats:
        subj = stat["subject"]
        sess = stat["session"]
        total_count += 1

        post = stat["regions"].get("occipital", stat["regions"].get("posterior", {}))
        d = post.get("cohens_d", 0)
        p = post.get("p_value", 1)
        ratio = post.get("ec_eo_ratio", 1)

        # Find corresponding classification result
        clf_acc = None
        for cr in clf_results:
            if cr["subject"] == subj and cr["session"] == sess:
                csp = cr["classifiers"].get("CSP_LDA", {})
                clf_acc = csp.get("accuracy_mean")
                break

        data_ok = d > 0.5 and p < 0.05
        clf_ok = clf_acc is not None and clf_acc > 0.7

        if data_ok:
            meaningful_count += 1
            verdict = "MEANINGFUL"
        else:
            verdict = "QUESTIONABLE"

        logger.info("  sub-%s ses-%s: %s", subj, sess, verdict)
        logger.info("    Alpha EC/EO ratio=%.2f, Cohen's d=%.2f, p=%.1e", ratio, d, p)
        if clf_acc is not None:
            logger.info("    CSP+LDA accuracy=%.1f%%", clf_acc * 100)
        logger.info("    Criteria: d>0.5 %s, p<0.05 %s, CSP>70%% %s",
                     "PASS" if d > 0.5 else "FAIL",
                     "PASS" if p < 0.05 else "FAIL",
                     "PASS" if clf_ok else ("FAIL" if clf_acc else "N/A"))

    logger.info("\n  Overall: %d/%d recordings show meaningful EEG signal",
                meaningful_count, total_count)
    if meaningful_count == total_count:
        logger.info("  CONCLUSION: EEG data appears physiologically valid.")
    elif meaningful_count > 0:
        logger.info("  CONCLUSION: Some recordings valid, some questionable — check session details.")
    else:
        logger.info("  CONCLUSION: EEG data does NOT show expected physiological patterns.")

    # ---- Save results ----
    results = {
        "alpha_power_stats": alpha_stats,
        "psd_summary": psd_stats,
        "classification": clf_results,
        "meaningful_recordings": meaningful_count,
        "total_recordings": total_count,
    }

    results_path = output_dir / "validation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("\nResults saved to %s", results_path)


if __name__ == "__main__":
    main()
