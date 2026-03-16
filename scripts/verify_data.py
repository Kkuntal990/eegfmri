"""Verify processed EEG-fMRI data integrity and correctness."""

import json
import logging
import sys

import numpy as np

sys.path.insert(0, "/projects/bbnv/kkokate/eegfmri")

from pathlib import Path
from eegfmri_loader import (
    discover_recordings,
    extract_events,
    load_dataset,
    load_raw,
)

logging.basicConfig(level=logging.WARNING)

root = Path("/work/hdd/bbnv/kuntal/eegfmri_data")
recs = discover_recordings(root)

# ===== 1: Metadata vs JSON sidecars =====
print("===== VERIFICATION 1: Metadata vs JSON sidecars =====")
for rec in recs:
    jp = rec["json_path"]
    if jp.exists():
        meta = json.loads(jp.read_text())
        dur = meta.get("RecordingDuration", 0)
        sfreq = meta.get("SamplingFrequency", 0)
        ch_count = meta.get("EEGChannelCount", 0) + meta.get("ECGChannelCount", 0)
        s, ses, t, r = rec["subject"], rec["session"], rec["task"], rec["run"] or "n/a"
        print(f"  sub-{s} ses-{ses} task-{t} run-{r}: "
              f"JSON says {sfreq} Hz, {dur:.1f}s, {ch_count} ch")

# ===== 2: Raw data integrity (spot check 4 files) =====
print("\n===== VERIFICATION 2: Raw data integrity (spot check) =====")
spot = [recs[0], recs[4], recs[9], recs[14]]  # 2 per subject, 1 ses-01 + 1 ses-02
for rec in spot:
    raw = load_raw(rec["set_path"])
    data = raw.get_data()
    s, ses, t = rec["subject"], rec["session"], rec["task"]
    print(f"  sub-{s} ses-{ses} task-{t}:")
    print(f"    Shape: {data.shape}, dtype: {data.dtype}")
    print(f"    NaN: {np.isnan(data).sum()}, Inf: {np.isinf(data).sum()}")
    print(f"    Range: [{data.min():.6f}, {data.max():.6f}] V")
    print(f"    Std (mean across ch): {data.std(axis=1).mean():.6f} V")

# ===== 3: Event counts per task =====
print("\n===== VERIFICATION 3: Event counts per recording =====")
for rec in recs:
    if not rec["events_path"].exists():
        continue
    raw = load_raw(rec["set_path"])
    events, event_id = extract_events(rec["events_path"], raw, rec["task"])
    s, ses, t, r = rec["subject"], rec["session"], rec["task"], rec["run"] or "n/a"
    if len(events) > 0:
        inv = {v: k for k, v in event_id.items()}
        counts = {}
        for eid in np.unique(events[:, 2]):
            counts[inv.get(eid, str(eid))] = int((events[:, 2] == eid).sum())
        print(f"  sub-{s} ses-{ses} task-{t} run-{r}: {len(events)} events -> {counts}")
    else:
        print(f"  sub-{s} ses-{ses} task-{t} run-{r}: 0 events")

# ===== 4: Preprocessed data sanity (ses-02 rest) =====
print("\n===== VERIFICATION 4: Preprocessed data sanity (ses-02 rest) =====")
dataset = load_dataset(root, tasks=["rest"], sessions=["02"])
for ds in dataset.datasets:
    raw = ds.raw
    data = raw.get_data()
    sub = ds.description["subject"]
    print(f"  sub-{sub}:")
    print(f"    Channels: {len(raw.ch_names)}, Sfreq: {raw.info['sfreq']} Hz")
    print(f"    Duration: {raw.times[-1]:.1f}s, Samples: {data.shape[1]}")
    print(f"    NaN: {np.isnan(data).sum()}, Inf: {np.isinf(data).sum()}")
    print(f"    Mean: {data.mean():.2e}, Std: {data.std():.2e}")
    print(f"    Min: {data.min():.2e}, Max: {data.max():.2e}")
    ch_var = data.var(axis=1)
    print(f"    Ch variance: min={ch_var.min():.2e}, max={ch_var.max():.2e}, "
          f"dead(var<1e-20)={int((ch_var < 1e-20).sum())}")

# ===== 5: Windowing check =====
print("\n===== VERIFICATION 5: Windowing shapes =====")
from braindecode.preprocessing import create_fixed_length_windows
windows = create_fixed_length_windows(
    dataset, window_size_samples=500, window_stride_samples=250, drop_last_window=True
)
print(f"  Total windows: {len(windows)}")
X, y, info = windows[0]
print(f"  Window shape: {X.shape}")
print(f"  Window dtype: {X.dtype}")
print(f"  Window range: [{X.min():.2e}, {X.max():.2e}]")
print(f"  Window NaN: {np.isnan(X).sum()}")

# Sample a few windows across the dataset
for idx in [0, len(windows) // 2, len(windows) - 1]:
    X, y, info = windows[idx]
    print(f"  Window[{idx}]: mean={X.mean():.2e}, std={X.std():.2e}")

print("\nAll verifications complete!")
