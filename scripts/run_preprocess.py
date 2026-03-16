"""Run full EEG-fMRI preprocessing pipeline and verify output."""

import argparse
import logging
import sys

sys.path.insert(0, "/projects/bbnv/kkokate/eegfmri")

from eegfmri_loader import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bids-root",
        default="/work/hdd/bbnv/kuntal/eegfmri_data",
        help="Path to BIDS dataset root",
    )
    parser.add_argument("--resample-freq", type=float, default=250.0)
    parser.add_argument("--no-mr-removal", action="store_true")
    parser.add_argument("--no-bcg-removal", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    print("Loading all recordings with preprocessing...")
    dataset = load_dataset(
        args.bids_root,
        preprocess=True,
        resample_freq=args.resample_freq,
        pick_eeg_only=True,
        mr_artifact_removal=not args.no_mr_removal,
        bcg_artifact_removal=not args.no_bcg_removal,
        n_jobs=args.n_jobs,
    )

    print(f"\nLoaded {len(dataset.datasets)} recordings")
    print(dataset.description)

    for i, ds in enumerate(dataset.datasets):
        raw = ds.raw
        print(
            f"  [{i}] sub-{ds.description['subject']} ses-{ds.description['session']} "
            f"task-{ds.description['task']} run-{ds.description['run']}: "
            f"{len(raw.ch_names)} ch, {raw.info['sfreq']} Hz, {raw.times[-1]:.1f}s"
        )

    # Windowing test
    from braindecode.preprocessing import create_fixed_length_windows

    windows = create_fixed_length_windows(
        dataset,
        window_size_samples=500,
        window_stride_samples=250,
        drop_last_window=True,
    )
    print(f"\nTotal windows: {len(windows)}")
    X, y, info = windows[0]
    print(f"Window shape: {X.shape}")
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
