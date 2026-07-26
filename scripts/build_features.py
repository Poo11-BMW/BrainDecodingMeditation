"""
Build the feature CSV from raw EEG BDF files.

LEAKAGE FIX: This script performs chronological time-splitting BEFORE
epoching, so no overlapping epoch can straddle a train/val/test boundary.
The `split` column in the output CSV encodes the partition for each epoch.

Usage:
    python scripts/build_features.py --config configs/default.yaml

Output:
    brain/rich_features.csv   — one row per epoch, with `split` column
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loading import bdf_path, list_subjects
from src.epoching import build_epoch_metadata, epoch_partition
from src.feature_extraction import extract_features
from src.splitting import chronological_time_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EEG feature CSV from raw BDF files.")
    p.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    p.add_argument("--subjects", nargs="*", default=None, help="Subset of subject IDs to process.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing rich_features.csv.")
    return p.parse_args()


def preprocess_raw(raw, cfg):
    """Apply standard EEG preprocessing pipeline."""
    import mne
    mne.set_log_level("ERROR")
    raw.pick_types(eeg=True, verbose=False)
    raw.filter(0.5, 40.0, method="fir", verbose=False)
    raw.notch_filter(50.0, method="fir", verbose=False)
    raw.resample(cfg.epoch.sfreq, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    # Per-channel z-score normalisation
    data = raw.get_data()
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
    raw._data = data
    return raw


def compute_psd_for_epoch(epoch_data: np.ndarray, sfreq: float, n_fft: int = 256):
    """Welch PSD for one epoch. Returns (psd, freqs)."""
    from scipy.signal import welch
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=min(n_fft, epoch_data.shape[-1]))
    return psd, freqs  # (n_channels, n_freqs), (n_freqs,)


def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)
    out_csv = Path(cfg.data_dir) / "rich_features.csv"

    if out_csv.exists() and not args.overwrite:
        logger.info("rich_features.csv already exists. Use --overwrite to rebuild.")
        return

    subjects = args.subjects or list_subjects(cfg.data_dir)
    if not subjects:
        logger.error("No subjects found in %s", cfg.data_dir)
        sys.exit(1)

    bands      = cfg.bands.as_dict()
    epoch_len  = cfg.epoch.epoch_len
    sfreq      = cfg.epoch.sfreq
    safety_gap = cfg.split.safety_gap_epochs * epoch_len

    all_rows: list[dict] = []

    for si, subject in enumerate(subjects):
        for task in cfg.tasks:
            bdf = bdf_path(cfg.data_dir, subject, task)
            if not bdf.exists():
                logger.warning("  Missing: %s", bdf)
                continue

            logger.info("[%02d/%d] %s / %s", si + 1, len(subjects), subject, task)

            try:
                import mne
                raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose=False)
                raw = preprocess_raw(raw, cfg)
            except Exception as exc:
                logger.error("  Failed to load %s: %s", bdf, exc)
                continue

            total_duration = raw.times[-1]  # seconds
            ch_names = [c.upper() for c in raw.ch_names]

            # ── Chronological split BEFORE epoching ──────────────────────────
            try:
                partitions = chronological_time_splits(
                    total_duration,
                    train_frac=cfg.split.train_frac,
                    val_frac=cfg.split.val_frac,
                    test_frac=cfg.split.test_frac,
                    safety_gap=safety_gap,
                )
            except ValueError as e:
                logger.warning("  Skipping %s/%s: %s", subject, task, e)
                del raw; gc.collect()
                continue

            for partition in partitions:
                epochs_data, start_times = epoch_partition(
                    raw, partition, epoch_len, sfreq
                )
                if len(epochs_data) == 0:
                    continue

                meta = build_epoch_metadata(
                    start_times, partition, epoch_len, subject, task,
                    recording_id=bdf.stem,
                )

                for ep_i, ep_start in enumerate(start_times):
                    sig = epochs_data[ep_i]          # (n_ch, n_times)
                    psd, freqs = compute_psd_for_epoch(sig, sfreq)

                    feat = extract_features(
                        epoch_data=sig,
                        psd_data=psd,
                        freqs=freqs,
                        ch_names=ch_names,
                        region_defs=cfg.region_defs,
                        bands=bands,
                        sfreq=sfreq,
                    )
                    feat.update({
                        "Subject":       subject,
                        "Task":          task,
                        "split":         partition.split,
                        "recording_id":  bdf.stem,
                        "start_s":       round(ep_start, 4),
                        "end_s":         round(ep_start + epoch_len, 4),
                    })
                    all_rows.append(feat)

            logger.info(
                "  %s/%s — %d epochs across train/val/test",
                subject, task, sum(len(p) for p in [partitions]),
            )
            del raw; gc.collect()

    if not all_rows:
        logger.error("No epochs extracted. Check data directory: %s", cfg.data_dir)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Saved → %s (%d rows × %d cols)", out_csv, len(df), df.shape[1])
    logger.info("Split distribution:\n%s", df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
