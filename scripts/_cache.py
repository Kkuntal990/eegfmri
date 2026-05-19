"""Disk caching for the two expensive preprocessing steps:

  1. EEG MR-gradient + BCG artifact removal (~90 s per run).
  2. ANTs rigid-body motion correction of BOLD (~60 s per run).

Cache layout (under /projects/bbnv/kkokate/eegfmri/derivatives/cache/):

  eeg_cleaned/sub-XXX/sub-XXX_ses-YY_task-T[_run-R]_cleaned-raw.fif
  bold_mc/sub-XXX/sub-XXX_ses-YY_task-T[_run-R]_mc.nii.gz
  bold_mc/sub-XXX/sub-XXX_ses-YY_task-T[_run-R]_fd.npy

All files are heavy and excluded from git via the `derivatives/` rule in
.gitignore. To rebuild from scratch, just `rm -rf derivatives/cache/`.
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

CACHE_ROOT = Path("/projects/bbnv/kkokate/eegfmri/derivatives/cache")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _eeg_fif_path(sub: str, ses: str, task: str, run: str | None,
                  root: Path = CACHE_ROOT) -> Path:
    run_part = f"_run-{run}" if run else ""
    return root / "eeg_cleaned" / f"sub-{sub}" / \
           f"sub-{sub}_ses-{ses}_task-{task}{run_part}_cleaned-raw.fif"


def _bold_paths(sub: str, ses: str, task: str, run: str | None,
                root: Path = CACHE_ROOT) -> tuple[Path, Path]:
    run_part = f"_run-{run}" if run else ""
    base = root / "bold_mc" / f"sub-{sub}"
    return (
        base / f"sub-{sub}_ses-{ses}_task-{task}{run_part}_mc.nii.gz",
        base / f"sub-{sub}_ses-{ses}_task-{task}{run_part}_fd.npy",
    )


def _discover_eeg_runs(bids_root: Path, sub: str, ses: str,
                       task: str) -> list[str | None]:
    """Return [None] for single-run task, ['1', '2', ...] for multi-run.

    Mirrors eegfmri_loader's BIDS regex so the cache keys line up with
    whatever load_dataset produces.
    """
    eeg_dir = bids_root / f"sub-{sub}" / f"ses-{ses}" / "eeg"
    pat = re.compile(
        rf"^sub-{re.escape(sub)}_ses-{re.escape(ses)}_task-{re.escape(task)}"
        r"(?:_run-(\d+))?_eeg\.set$"
    )
    runs = []
    for f in sorted(eeg_dir.glob(f"sub-{sub}_ses-{ses}_task-{task}*_eeg.set")):
        m = pat.match(f.name)
        if m:
            runs.append(m.group(1))  # None or "1"/"2"/...
    return runs or [None]


# ---------------------------------------------------------------------------
# EEG cache: full task at once (load_dataset operates per-task)
# ---------------------------------------------------------------------------

def get_cleaned_eeg_for_task(bids_root: Path, sub: str, ses: str, task: str,
                             cache_root: Path = CACHE_ROOT,
                             force: bool = False) -> dict[str | None, "mne.io.Raw"]:
    """Return ``{run_id_or_None: MNE Raw}`` for one subject/session/task,
    loading from cache if every expected run is already cached.

    On cache miss we run the full eegfmri_loader pipeline once (which is
    cheaper than per-run invocations because BCG R-peak detection etc.
    don't need to be re-bootstrapped) and persist each cleaned Raw as .fif.
    """
    import mne

    expected_runs = _discover_eeg_runs(bids_root, sub, ses, task)
    paths = {r: _eeg_fif_path(sub, ses, task, r, cache_root) for r in expected_runs}

    if not force and all(p.exists() for p in paths.values()):
        log.info("EEG cache HIT: sub-%s task-%s runs=%s", sub, task,
                 list(paths.keys()))
        return {r: mne.io.read_raw_fif(str(p), preload=True, verbose=False)
                for r, p in paths.items()}

    log.info("EEG cache MISS: sub-%s task-%s -- running cleanup", sub, task)
    sys.path.insert(0, "/projects/bbnv/kkokate/eegfmri")
    from eegfmri_loader import load_dataset
    ds = load_dataset(
        bids_root, subjects=[sub], sessions=[ses], tasks=[task],
        preprocess=True, resample_freq=250.0, pick_eeg_only=True,
        mr_artifact_removal=True, bcg_artifact_removal=True, n_jobs=1,
    )

    out: dict[str | None, "mne.io.Raw"] = {}
    for d in ds.datasets:
        r = d.description.get("run")
        # Normalise '1.0', 1, '1' -> '1'; None stays None.
        if r is None or (isinstance(r, float) and np.isnan(r)):
            r_key = None
        else:
            try:
                r_key = str(int(r))
            except (ValueError, TypeError):
                r_key = str(r) if str(r) else None
        out[r_key] = d.raw
        p = paths.get(r_key)
        if p is None:
            # Run wasn't pre-discovered; build a path on the fly.
            p = _eeg_fif_path(sub, ses, task, r_key, cache_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        d.raw.save(str(p), overwrite=True, verbose=False)
        log.info("  saved %s", p.name)
    return out


# ---------------------------------------------------------------------------
# BOLD cache: per-file ANTs motion correction
# ---------------------------------------------------------------------------

def get_motion_corrected_bold(bold_path: Path, sub: str, ses: str, task: str,
                              run: str | None = None,
                              cache_root: Path = CACHE_ROOT,
                              force: bool = False) -> tuple["nib.Nifti1Image", np.ndarray]:
    """Return ``(motion_corrected_image, FD_array)``, caching on disk."""
    nii_path, fd_path = _bold_paths(sub, ses, task, run, cache_root)
    if not force and nii_path.exists() and fd_path.exists():
        log.info("BOLD cache HIT: %s", nii_path.name)
        return nib.load(str(nii_path)), np.load(str(fd_path))

    log.info("BOLD cache MISS: motion-correcting %s", bold_path.name)
    import ants
    ants_img = ants.image_read(str(bold_path))
    mc = ants.motion_correction(ants_img)

    # Write to a temp file first so we don't leave a half-written cache entry
    # if the process dies mid-write.
    nii_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            suffix=".nii.gz", dir=str(nii_path.parent), delete=False) as f:
        tmp_path = f.name
    ants.image_write(mc["motion_corrected"], tmp_path)
    Path(tmp_path).replace(nii_path)

    fd = np.asarray(mc.get("FD", []), dtype=float)
    if fd.size == 0:
        # Fallback: zero-fill to match nii length.
        n_trs = nib.load(str(nii_path)).shape[3]
        fd = np.zeros(n_trs, dtype=float)
    np.save(str(fd_path), fd)
    log.info("  saved %s + %s", nii_path.name, fd_path.name)
    return nib.load(str(nii_path)), fd
