"""
Epoch generation from raw MNE objects, partition-aware.

Epochs are generated WITHIN each temporal partition so no epoch can
straddle a train/val/test boundary.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

from src.splitting import TimePartition

logger = logging.getLogger(__name__)


def epoch_partition(
    raw: "mne.io.BaseRaw",
    partition: TimePartition,
    epoch_len: float,
    sfreq: int,
) -> tuple[np.ndarray, list[float]]:
    """
    Extract non-overlapping epochs from `raw` within a time partition.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw object (must already be filtered/resampled/referenced).
    partition : TimePartition
        The time block to epoch ([start, end) in seconds).
    epoch_len : float
        Epoch length in seconds.
    sfreq : int
        Expected sampling frequency (validated against raw.info["sfreq"]).

    Returns
    -------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
    epoch_start_times : list[float]
        Start time in seconds for each epoch (for metadata).
    """
    actual_sfreq = raw.info["sfreq"]
    if abs(actual_sfreq - sfreq) > 0.5:
        raise ValueError(
            f"Raw sfreq={actual_sfreq} does not match expected sfreq={sfreq}. "
            "Resample before epoching."
        )

    n_times_per_epoch = int(epoch_len * sfreq)
    start_sample = int(partition.start * sfreq)
    end_sample   = int(partition.end   * sfreq)

    data = raw.get_data()  # (n_channels, n_total_times)
    total_samples = data.shape[1]
    end_sample = min(end_sample, total_samples)

    epochs_data: list[np.ndarray] = []
    epoch_starts: list[float] = []

    t = start_sample
    while t + n_times_per_epoch <= end_sample:
        epoch = data[:, t: t + n_times_per_epoch]
        epochs_data.append(epoch)
        epoch_starts.append(t / sfreq)
        t += n_times_per_epoch  # non-overlapping step

    if not epochs_data:
        logger.warning(
            "No epochs extracted from partition [%.1f, %.1f) — partition too short?",
            partition.start, partition.end,
        )
        return np.empty((0, data.shape[0], n_times_per_epoch)), []

    return np.stack(epochs_data, axis=0), epoch_starts


def build_epoch_metadata(
    epoch_start_times: list[float],
    partition: TimePartition,
    epoch_len: float,
    subject: str,
    task: str,
    recording_id: str = "",
) -> pd.DataFrame:
    """Build a metadata DataFrame for a set of epochs."""
    records = []
    for t in epoch_start_times:
        records.append({
            "subject":      subject,
            "task":         task,
            "recording_id": recording_id,
            "split":        partition.split,
            "start_s":      round(t, 6),
            "end_s":        round(t + epoch_len, 6),
        })
    return pd.DataFrame(records)
