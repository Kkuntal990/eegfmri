"""
EEG-fMRI data loader for braindecode.

Loads simultaneous EEG-fMRI data from BIDS-formatted EEGLAB .set/.fdt files
into braindecode BaseConcatDataset for deep learning analysis.

Usage:
    from eegfmri_loader import load_dataset
    dataset = load_dataset('/path/to/bids_root', tasks=['rest', 'swm'])
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Generate missing BIDS root files
# ---------------------------------------------------------------------------

def generate_bids_root(bids_root: str | Path) -> None:
    """Create minimal BIDS compliance files if missing."""
    bids_root = Path(bids_root)

    # dataset_description.json
    desc_path = bids_root / "dataset_description.json"
    if not desc_path.exists():
        desc = {
            "Name": "EEG-fMRI Dataset",
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
        }
        desc_path.write_text(json.dumps(desc, indent=2) + "\n")
        logger.info("Created %s", desc_path)

    # participants.tsv
    participants_path = bids_root / "participants.tsv"
    if not participants_path.exists():
        sub_dirs = sorted(
            d.name for d in bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
        lines = ["participant_id"]
        lines.extend(sub_dirs)
        participants_path.write_text("\n".join(lines) + "\n")
        logger.info("Created %s with %d subjects", participants_path, len(sub_dirs))


# ---------------------------------------------------------------------------
# Step 2a: Discover recordings
# ---------------------------------------------------------------------------

_BIDS_RE = re.compile(
    r"sub-(?P<subject>\w+)_ses-(?P<session>\w+)_task-(?P<task>\w+)"
    r"(?:_run-(?P<run>\d+))?_eeg\.set$"
)


def discover_recordings(
    bids_root: str | Path,
    subjects: Optional[list[str]] = None,
    sessions: Optional[list[str]] = None,
    tasks: Optional[list[str]] = None,
    runs: Optional[list[str]] = None,
) -> list[dict]:
    """Discover all EEG recordings in the BIDS tree.

    Returns a list of dicts with keys: subject, session, task, run,
    set_path, fdt_path, events_path, json_path, channels_path.
    """
    bids_root = Path(bids_root)
    recordings = []

    for set_path in sorted(bids_root.glob("sub-*/ses-*/eeg/*_eeg.set")):
        m = _BIDS_RE.search(set_path.name)
        if m is None:
            continue

        rec = {
            "subject": m.group("subject"),
            "session": m.group("session"),
            "task": m.group("task"),
            "run": m.group("run"),
            "set_path": set_path,
        }

        # Apply filters
        if subjects and rec["subject"] not in subjects:
            continue
        if sessions and rec["session"] not in sessions:
            continue
        if tasks and rec["task"] not in tasks:
            continue
        if runs and rec["run"] not in runs:
            continue

        # Resolve associated sidecar files
        stem = set_path.name.replace("_eeg.set", "")
        eeg_dir = set_path.parent
        ses_dir = eeg_dir.parent

        rec["fdt_path"] = set_path.with_suffix(".fdt")
        rec["events_path"] = eeg_dir / f"{stem}_events.tsv"
        rec["json_path"] = eeg_dir / f"{stem}_eeg.json"
        # channels.tsv is per-session, not per-task
        channels_name = f"sub-{rec['subject']}_ses-{rec['session']}_channels.tsv"
        rec["channels_path"] = eeg_dir / channels_name

        recordings.append(rec)

    logger.info("Discovered %d recordings in %s", len(recordings), bids_root)
    return recordings


# ---------------------------------------------------------------------------
# Step 2b: Load raw EEG
# ---------------------------------------------------------------------------

def load_raw(set_path: str | Path) -> mne.io.Raw:
    """Load a single EEGLAB .set file and configure channel types/montage."""
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True)
    except ValueError as e:
        if "inhomogeneous" not in str(e):
            raise
        # Numpy >= 2.0 can fail on .set files with inconsistent channel
        # coordinates. Temporarily patch np.atleast_1d to handle this,
        # since we set our own montage anyway.
        logger.warning("Retrying load with numpy inhomogeneous-array workaround: %s", set_path)
        _orig = np.atleast_1d
        def _safe_atleast_1d(ary):
            try:
                return _orig(ary)
            except ValueError:
                return np.asarray(ary, dtype=object)
        np.atleast_1d = _safe_atleast_1d
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True)
        finally:
            np.atleast_1d = _orig

    # Set channel types: E1-E129 -> eeg, ECG -> ecg
    ch_types = {}
    for ch in raw.ch_names:
        if ch.upper() == "ECG":
            ch_types[ch] = "ecg"
        elif re.match(r"^E\d+$", ch):
            ch_types[ch] = "eeg"
    if ch_types:
        raw.set_channel_types(ch_types)

    # E129 is the Cz reference electrode — rename to match montage convention
    if "E129" in raw.ch_names:
        raw.rename_channels({"E129": "Cz"})

    # Apply standard EGI HydroCel 129 montage to EEG channels
    try:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        raw.set_montage(montage, on_missing="warn")
    except Exception as e:
        logger.warning("Could not set montage: %s", e)

    return raw


# ---------------------------------------------------------------------------
# Step 2c: Preprocessing
# ---------------------------------------------------------------------------

def _remove_mr_gradient_artifact(raw: mne.io.Raw, n_neighbors: int = 25) -> mne.io.Raw:
    """Remove MR gradient artifact using Average Artifact Subtraction (AAS).

    Uses TREV annotations as volume markers. For each TR interval, builds a
    template from neighboring TRs and subtracts it.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with TREV annotations present.
    n_neighbors : int
        Number of neighboring TRs on each side for template averaging.

    Returns
    -------
    mne.io.Raw
        Raw data with gradient artifact removed.
    """
    # Find TREV onset samples from annotations
    trev_onsets = []
    for ann in raw.annotations:
        if ann["description"] == "TREV":
            trev_onsets.append(int(ann["onset"] * raw.info["sfreq"]))
    trev_onsets = np.array(sorted(trev_onsets))

    if len(trev_onsets) < 3:
        logger.warning("Too few TREV events (%d) for MR artifact removal", len(trev_onsets))
        return raw

    # Compute TR lengths (samples between consecutive TREVs)
    tr_lengths = np.diff(trev_onsets)
    median_tr = int(np.median(tr_lengths))
    logger.info(
        "MR artifact removal: %d TRs, median TR = %d samples (%.3f s)",
        len(trev_onsets), median_tr, median_tr / raw.info["sfreq"],
    )

    data = raw.get_data()  # (n_channels, n_samples)
    n_channels, n_samples = data.shape

    for i in range(len(trev_onsets)):
        # Define this TR segment
        start = trev_onsets[i]
        end = start + median_tr
        if end > n_samples:
            break

        seg_len = end - start

        # Gather neighboring TR segments for template
        neighbors = []
        for j in range(max(0, i - n_neighbors), min(len(trev_onsets), i + n_neighbors + 1)):
            if j == i:
                continue
            n_start = trev_onsets[j]
            n_end = n_start + seg_len
            if n_end <= n_samples:
                neighbors.append(data[:, n_start:n_end])

        if len(neighbors) < 3:
            continue

        # Build template as mean of neighbors and subtract
        template = np.mean(neighbors, axis=0)
        data[:, start:end] -= template

    # Write corrected data back
    raw._data = data
    logger.info("MR gradient artifact removal complete")
    return raw


def _remove_bcg_artifact(raw: mne.io.Raw, n_neighbors: int = 25) -> mne.io.Raw:
    """Remove BCG (ballistocardiogram) artifact using AAS on ECG-locked epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with ECG channel present (after MR artifact removal).
    n_neighbors : int
        Number of neighboring heartbeats for template averaging.

    Returns
    -------
    mne.io.Raw
        Raw data with BCG artifact removed.
    """
    # Find ECG channel
    ecg_chs = mne.pick_types(raw.info, ecg=True)
    if len(ecg_chs) == 0:
        logger.warning("No ECG channel found, skipping BCG artifact removal")
        return raw

    # Detect R-peaks
    ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw)
    if len(ecg_events) < 5:
        logger.warning("Too few ECG events (%d) for BCG removal", len(ecg_events))
        return raw

    r_peaks = ecg_events[:, 0]  # sample indices
    logger.info("BCG artifact removal: %d R-peaks detected", len(r_peaks))

    # Compute median heartbeat interval
    hb_intervals = np.diff(r_peaks)
    median_hb = int(np.median(hb_intervals))

    # Define epoch window centered on R-peak
    pre_samples = median_hb // 2
    post_samples = median_hb - pre_samples

    data = raw.get_data()
    n_channels, n_samples = data.shape

    # Pick only EEG channels for subtraction
    eeg_picks = mne.pick_types(raw.info, eeg=True, ecg=False)

    for i in range(len(r_peaks)):
        start = r_peaks[i] - pre_samples
        end = r_peaks[i] + post_samples
        if start < 0 or end > n_samples:
            continue

        seg_len = end - start

        # Gather neighboring heartbeat segments
        neighbors = []
        for j in range(max(0, i - n_neighbors), min(len(r_peaks), i + n_neighbors + 1)):
            if j == i:
                continue
            n_start = r_peaks[j] - pre_samples
            n_end = r_peaks[j] + post_samples
            if n_start >= 0 and n_end <= n_samples and (n_end - n_start) == seg_len:
                neighbors.append(data[np.ix_(eeg_picks, range(n_start, n_end))])

        if len(neighbors) < 3:
            continue

        template = np.mean(neighbors, axis=0)
        data[np.ix_(eeg_picks, range(start, end))] -= template

    raw._data = data
    logger.info("BCG artifact removal complete")
    return raw


def preprocess_raw(
    raw: mne.io.Raw,
    session: str,
    resample_freq: float = 250.0,
    l_freq: float = 0.1,
    h_freq: float = 100.0,
    notch: float = 60.0,
    pick_eeg: bool = True,
    mr_artifact_removal: bool = True,
    bcg_artifact_removal: bool = True,
) -> mne.io.Raw:
    """Preprocess raw EEG data.

    For ses-02 data (recorded inside scanner), applies MR gradient and BCG
    artifact removal before standard preprocessing.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (preloaded).
    session : str
        Session identifier (e.g. '01', '02'). Artifact removal only for '02'.
    resample_freq : float
        Target sampling frequency in Hz.
    l_freq, h_freq : float
        Bandpass filter edges in Hz.
    notch : float
        Notch filter frequency (power line).
    pick_eeg : bool
        If True, drop non-EEG channels after artifact removal.
    mr_artifact_removal : bool
        Apply MR gradient artifact removal (ses-02 only).
    bcg_artifact_removal : bool
        Apply BCG artifact removal (ses-02 only, requires ECG channel).

    Returns
    -------
    mne.io.Raw
        Preprocessed raw data.
    """
    is_scanner_session = session in ("02", "ses-02")

    # MR gradient artifact removal (ses-02 only)
    if is_scanner_session and mr_artifact_removal:
        logger.info("Applying MR gradient artifact removal (session=%s)", session)
        raw = _remove_mr_gradient_artifact(raw)

    # BCG artifact removal (ses-02 only, after MR correction)
    if is_scanner_session and bcg_artifact_removal:
        logger.info("Applying BCG artifact removal (session=%s)", session)
        raw = _remove_bcg_artifact(raw)

    # Drop non-EEG channels
    if pick_eeg:
        raw.pick("eeg")

    # Bandpass filter
    raw.filter(l_freq, h_freq, fir_design="firwin")

    # Notch filter (power line)
    raw.notch_filter(notch, fir_design="firwin")

    # Resample
    if resample_freq and raw.info["sfreq"] != resample_freq:
        raw.resample(resample_freq)

    return raw


# ---------------------------------------------------------------------------
# Step 2d: Event extraction
# ---------------------------------------------------------------------------

# Events to filter out (scanner triggers and block markers)
_SKIP_EVENTS = {"TREV", "START", "boundary"}
_BLOCK_MARKERS = {"ECstart", "EOstart"}

# Task-specific event ID mappings
_TASK_EVENT_IDS = {
    "rest": {"EO": 1},
    "eoec": {"EC": 1, "EO": 2},
    "stimloc": {"loc_targ_l1": 1, "loc_targ_l5": 2, "loc_redcross": 3},
    "cpt": {},  # built dynamically from CPT_ prefixes
    "swm": {},  # built dynamically from SDRT_ prefixes
}


def extract_events(
    events_tsv_path: str | Path,
    raw: mne.io.Raw,
    task: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract MNE-format events array from a BIDS events.tsv file.

    Parameters
    ----------
    events_tsv_path : str | Path
        Path to the _events.tsv sidecar.
    raw : mne.io.Raw
        The corresponding Raw object (for sampling frequency).
    task : str
        Task name for task-specific event mapping.

    Returns
    -------
    events : np.ndarray, shape (n_events, 3)
        MNE events array [sample, 0, event_id].
    event_id : dict
        Mapping from event label to integer event ID.
    """
    df = pd.read_csv(events_tsv_path, sep="\t")

    if "onset" not in df.columns or "value" not in df.columns:
        logger.warning("Missing onset/value columns in %s", events_tsv_path)
        return np.empty((0, 3), dtype=int), {}

    sfreq = raw.info["sfreq"]

    # Filter out scanner triggers and block start markers
    skip = _SKIP_EVENTS | _BLOCK_MARKERS
    mask = ~df["value"].astype(str).isin(skip)
    df = df[mask].copy()

    if df.empty:
        return np.empty((0, 3), dtype=int), {}

    # Build event_id mapping based on task
    unique_values = sorted(df["value"].astype(str).unique())

    if task == "rest":
        event_id = {"EO": 1}
    elif task == "eoec":
        event_id = {"EC": 1, "EO": 2}
    elif task == "stimloc":
        event_id = {v: i + 1 for i, v in enumerate(
            sorted(v for v in unique_values if v.startswith("loc_"))
        )}
    elif task == "cpt":
        # Group CPT events: go (O) vs nogo (X), plus responses
        go_stims = sorted(v for v in unique_values if re.match(r"CPT_O\d+corr", v))
        nogo_stims = sorted(v for v in unique_values if re.match(r"CPT_X\d+corr", v))
        responses = sorted(v for v in unique_values if "Resp" in v)
        event_id = {}
        idx = 1
        for v in go_stims:
            event_id[v] = idx
            idx += 1
        for v in nogo_stims:
            event_id[v] = idx
            idx += 1
        for v in responses:
            event_id[v] = idx
            idx += 1
    elif task == "swm":
        # Group SWM events by phase and load
        event_id = {}
        idx = 1
        for v in unique_values:
            if v.startswith("SDRT_"):
                event_id[v] = idx
                idx += 1
    else:
        # Generic mapping for unknown tasks (e.g. Dataset 2 face tasks)
        event_id = {v: i + 1 for i, v in enumerate(unique_values)}

    # Build events array: compute sample from onset (don't rely on sample column)
    events_list = []
    for _, row in df.iterrows():
        label = str(row["value"])
        if label not in event_id:
            continue
        sample = int(round(float(row["onset"]) * sfreq))
        events_list.append([sample, 0, event_id[label]])

    if not events_list:
        return np.empty((0, 3), dtype=int), {}

    events = np.array(events_list, dtype=int)
    # Sort by sample
    events = events[events[:, 0].argsort()]

    logger.info(
        "Extracted %d events for task=%s from %s",
        len(events), task, events_tsv_path,
    )
    return events, event_id


# ---------------------------------------------------------------------------
# Step 2e: Create braindecode dataset
# ---------------------------------------------------------------------------

def create_braindecode_dataset(
    raw: mne.io.Raw,
    subject: str,
    session: str,
    task: str,
    run: Optional[str],
    events: np.ndarray,
    event_id: dict[str, int],
):
    """Wrap a preprocessed Raw object as a braindecode BaseDataset.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data.
    subject, session, task, run : str
        BIDS metadata.
    events : np.ndarray
        MNE events array.
    event_id : dict
        Event label -> ID mapping.

    Returns
    -------
    braindecode.datasets.BaseDataset
    """
    from braindecode.datasets import RawDataset

    # Attach events as annotations on the Raw object
    if len(events) > 0:
        inv_event_id = {v: k for k, v in event_id.items()}
        onsets = events[:, 0] / raw.info["sfreq"]
        durations = np.zeros(len(events))
        descriptions = [inv_event_id.get(eid, str(eid)) for eid in events[:, 2]]
        annotations = mne.Annotations(onsets, durations, descriptions)
        raw.set_annotations(annotations)

    # Build description
    description = pd.Series({
        "subject": subject,
        "session": session,
        "task": task,
        "run": run if run else "n/a",
    })

    return RawDataset(raw, description)


# ---------------------------------------------------------------------------
# Step 2f: Main entry point
# ---------------------------------------------------------------------------

def _process_one_recording(
    rec: dict,
    preprocess: bool,
    resample_freq: float,
    pick_eeg_only: bool,
    mr_artifact_removal: bool,
    bcg_artifact_removal: bool,
) -> object:
    """Process a single recording: load -> preprocess -> events -> BaseDataset."""
    logger.info(
        "Processing sub-%s ses-%s task-%s run-%s",
        rec["subject"], rec["session"], rec["task"], rec.get("run", "n/a"),
    )

    raw = load_raw(rec["set_path"])

    if preprocess:
        raw = preprocess_raw(
            raw,
            session=rec["session"],
            resample_freq=resample_freq,
            pick_eeg=pick_eeg_only,
            mr_artifact_removal=mr_artifact_removal,
            bcg_artifact_removal=bcg_artifact_removal,
        )

    # Extract events
    events = np.empty((0, 3), dtype=int)
    event_id = {}
    if rec["events_path"].exists():
        events, event_id = extract_events(rec["events_path"], raw, rec["task"])

    ds = create_braindecode_dataset(
        raw, rec["subject"], rec["session"], rec["task"], rec["run"],
        events, event_id,
    )
    return ds


def load_dataset(
    bids_root: str | Path,
    subjects: Optional[list[str]] = None,
    sessions: Optional[list[str]] = None,
    tasks: Optional[list[str]] = None,
    runs: Optional[list[str]] = None,
    preprocess: bool = True,
    resample_freq: float = 250.0,
    pick_eeg_only: bool = True,
    mr_artifact_removal: bool = True,
    bcg_artifact_removal: bool = True,
    n_jobs: int = 1,
):
    """Load EEG-fMRI data into a braindecode BaseConcatDataset.

    This is the main entry point. It discovers recordings, loads and
    preprocesses each one, extracts events, and returns a concatenated
    dataset ready for windowing and model training.

    Parameters
    ----------
    bids_root : str | Path
        Path to the BIDS dataset root.
    subjects : list of str, optional
        Subject IDs to include (e.g. ['500', '1070302']).
    sessions : list of str, optional
        Sessions to include (e.g. ['01', '02']).
    tasks : list of str, optional
        Tasks to include (e.g. ['rest', 'swm', 'eoec']).
    runs : list of str, optional
        Runs to include (e.g. ['1', '2']).
    preprocess : bool
        Whether to apply preprocessing pipeline.
    resample_freq : float
        Target sampling frequency after resampling.
    pick_eeg_only : bool
        Drop non-EEG channels after artifact removal.
    mr_artifact_removal : bool
        Apply MR gradient artifact removal for ses-02 data.
    bcg_artifact_removal : bool
        Apply BCG artifact removal for ses-02 data.
    n_jobs : int
        Number of parallel jobs (via joblib).

    Returns
    -------
    braindecode.datasets.BaseConcatDataset
        Concatenated dataset with one BaseDataset per recording.

    Examples
    --------
    >>> dataset = load_dataset('/data/eegfmri', tasks=['rest'], sessions=['02'])
    >>> print(len(dataset.datasets))  # number of recordings
    >>> print(dataset.description)    # metadata DataFrame

    # Create fixed-length windows for rest data:
    >>> from braindecode.preprocessing import create_fixed_length_windows
    >>> windows = create_fixed_length_windows(dataset, window_size_samples=500)

    # Create event-locked windows for task data:
    >>> from braindecode.preprocessing import create_windows_from_events
    >>> windows = create_windows_from_events(dataset)
    """
    from braindecode.datasets import BaseConcatDataset

    bids_root = Path(bids_root)

    # Ensure BIDS root files exist
    generate_bids_root(bids_root)

    # Discover recordings
    recordings = discover_recordings(bids_root, subjects, sessions, tasks, runs)
    if not recordings:
        raise FileNotFoundError(
            f"No EEG recordings found in {bids_root} with the given filters"
        )

    # Process recordings (optionally in parallel)
    if n_jobs == 1:
        datasets = [
            _process_one_recording(
                rec, preprocess, resample_freq, pick_eeg_only,
                mr_artifact_removal, bcg_artifact_removal,
            )
            for rec in recordings
        ]
    else:
        datasets = Parallel(n_jobs=n_jobs)(
            delayed(_process_one_recording)(
                rec, preprocess, resample_freq, pick_eeg_only,
                mr_artifact_removal, bcg_artifact_removal,
            )
            for rec in recordings
        )

    return BaseConcatDataset(datasets)


# ---------------------------------------------------------------------------
# CLI: quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Discovering recordings in: {root}")
    recs = discover_recordings(root)
    for r in recs:
        print(f"  sub-{r['subject']} ses-{r['session']} task-{r['task']} "
              f"run-{r['run'] or 'n/a'}  ->  {r['set_path'].name}")
    print(f"\nTotal: {len(recs)} recordings")
