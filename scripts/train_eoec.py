"""Train ShallowFBCSPNet on Eyes Open vs Eyes Closed classification.

Binary classification: EC (eyes closed) = 0, EO (eyes open) = 1.
Uses leave-one-subject-out cross-validation (2 subjects).
Loads both ses-01 (outside scanner) and ses-02 (inside scanner, artifact-corrected).

Uses ShallowFBCSPNet (fewer parameters, works well with small EEG datasets)
with early stopping and class-weighted loss.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import mne
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, "/projects/bbnv/kkokate/eegfmri")

from eegfmri_loader import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_MAP = {"EC": 0, "EO": 1}


def extract_epochs(dataset, window_size_samples=500):
    """Extract numpy arrays from dataset using MNE Epochs.

    Returns X (n_windows, n_chans, n_times), y (n_windows,), subjects (n_windows,).
    """
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    tmax = (window_size_samples - 1) / sfreq

    all_X, all_y, all_subjects = [], [], []

    for ds in dataset.datasets:
        raw = ds.raw
        subject = ds.description["subject"]

        events, event_id = mne.events_from_annotations(raw)

        keep_ids = {k: v for k, v in event_id.items() if k in TARGET_MAP}
        if not keep_ids:
            logger.warning("No EC/EO events for sub-%s ses-%s, skipping",
                           subject, ds.description["session"])
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
                     subject, ds.description["session"],
                     len(y), (y == 0).sum(), (y == 1).sum())

        all_X.append(X)
        all_y.append(y)
        all_subjects.extend([subject] * len(y))

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subjects = np.array(all_subjects)

    return X, y, subjects


def drop_cz_channel(dataset):
    """Drop the Cz reference channel (all zeros) from all recordings."""
    for ds in dataset.datasets:
        if "Cz" in ds.raw.ch_names:
            ds.raw.drop_channels(["Cz"])


def build_model(model_name, n_chans, n_times, sfreq=250, lr=1e-3, weight_decay=1e-2,
                n_epochs=200, batch_size=64, patience=20, device="cuda",
                class_weights=None):
    """Build EEG classifier with skorch."""
    from braindecode.models import ShallowFBCSPNet, EEGNetv4
    from braindecode import EEGClassifier
    from skorch.callbacks import LRScheduler, EpochScoring, EarlyStopping
    from skorch.dataset import ValidSplit

    if model_name == "ShallowFBCSPNet":
        model = ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
        )
    elif model_name == "EEGNetv4":
        model = EEGNetv4(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Class-weighted cross-entropy loss
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = torch.nn.CrossEntropyLoss

    callbacks = [
        ("train_acc", EpochScoring(
            "accuracy", on_train=True, name="train_acc", lower_is_better=False)),
        ("early_stopping", EarlyStopping(
            monitor="valid_loss", patience=patience, lower_is_better=True)),
    ]

    clf = EEGClassifier(
        model,
        criterion=criterion,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=n_epochs,
        train_split=ValidSplit(cv=0.2, stratified=True),
        callbacks=callbacks,
        device=device,
        verbose=1,
    )

    return clf


def evaluate(clf, X_test, y_test):
    """Evaluate classifier and return metrics dict."""
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["EC", "EO"])

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test": len(y_test),
        "class_counts": {
            "EC": int((y_test == 0).sum()),
            "EO": int((y_test == 1).sum()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train EOEC classifier")
    parser.add_argument("--bids-root", default="/work/hdd/bbnv/kuntal/eegfmri_data")
    parser.add_argument("--output-dir", default="/projects/bbnv/kkokate/eegfmri/results/eoec")
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Sessions to use (e.g. 01 02). Default: both.")
    parser.add_argument("--model", default="ShallowFBCSPNet",
                        choices=["ShallowFBCSPNet", "EEGNetv4"],
                        help="Model architecture (default: ShallowFBCSPNet)")
    parser.add_argument("--window-size", type=int, default=500,
                        help="Window size in samples (default: 500 = 2s at 250Hz)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
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
    print(dataset.description)

    # Drop flat Cz reference channel
    drop_cz_channel(dataset)
    n_chans = len(dataset.datasets[0].raw.ch_names)
    logger.info("Channels after dropping Cz: %d", n_chans)

    # ---- Extract epochs as numpy arrays ----
    logger.info("Extracting epochs (window_size=%d samples)", args.window_size)
    X, y, subjects = extract_epochs(dataset, window_size_samples=args.window_size)
    logger.info("Total epochs: %d, shape: %s", len(y), X.shape)
    logger.info("Class distribution: EC=%d, EO=%d", (y == 0).sum(), (y == 1).sum())

    # ---- Get unique subjects ----
    unique_subjects = sorted(set(subjects))
    logger.info("Subjects: %s", unique_subjects)

    if len(unique_subjects) < 2:
        raise ValueError(f"Need >= 2 subjects for LOSO CV, got {len(unique_subjects)}")

    # ---- Leave-one-subject-out cross-validation ----
    all_results = {}

    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        logger.info("=" * 60)
        logger.info("LOSO fold: test=%s", test_subj)
        logger.info("  Train: %d epochs (EC=%d, EO=%d)",
                     len(y_train), (y_train == 0).sum(), (y_train == 1).sum())
        logger.info("  Test:  %d epochs (EC=%d, EO=%d)",
                     len(y_test), (y_test == 0).sum(), (y_test == 1).sum())
        logger.info("=" * 60)

        # Compute class weights for this fold's training data
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        logger.info("Class weights: EC=%.3f, EO=%.3f", cw[0], cw[1])

        # Build and train model
        clf = build_model(
            model_name=args.model,
            n_chans=n_chans,
            n_times=args.window_size,
            lr=args.lr,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
            class_weights=cw,
        )

        clf.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(clf, X_test, y_test)
        metrics["test_subject"] = test_subj
        metrics["model"] = args.model
        metrics["stopped_epoch"] = len(clf.history)
        all_results[f"test_{test_subj}"] = metrics

        logger.info("Test sub-%s: accuracy=%.4f, balanced_accuracy=%.4f (stopped at epoch %d)",
                     test_subj, metrics["accuracy"], metrics["balanced_accuracy"],
                     metrics["stopped_epoch"])
        print(metrics["classification_report"])
        print("Confusion matrix:")
        print(np.array(metrics["confusion_matrix"]))

        # Save fold model
        torch.save(clf.module_.state_dict(), output_dir / f"model_{args.model}_test_{test_subj}.pt")

    # ---- Aggregate results ----
    mean_acc = np.mean([r["accuracy"] for r in all_results.values()])
    mean_bal_acc = np.mean([r["balanced_accuracy"] for r in all_results.values()])

    summary = {
        "mean_accuracy": float(mean_acc),
        "mean_balanced_accuracy": float(mean_bal_acc),
        "folds": all_results,
        "config": {
            "sessions": args.sessions,
            "window_size": args.window_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "n_chans": n_chans,
            "model": args.model,
        },
    }

    results_path = output_dir / f"results_{args.model}.json"
    results_path.write_text(json.dumps(summary, indent=2))
    logger.info("Results saved to %s", results_path)

    logger.info("=" * 60)
    logger.info("LOSO SUMMARY (%s)", args.model)
    logger.info("  Mean accuracy:          %.4f", mean_acc)
    logger.info("  Mean balanced accuracy:  %.4f", mean_bal_acc)
    for fold, r in all_results.items():
        logger.info("  %s: acc=%.4f bal_acc=%.4f (epoch %d)",
                     fold, r["accuracy"], r["balanced_accuracy"], r["stopped_epoch"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
