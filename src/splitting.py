"""
Chronological train/val/test splitting that prevents overlapping-epoch leakage.

The key invariant: raw recording time is split BEFORE epochs are generated.
Each partition is epoched independently.  No epoch ever straddles a boundary.
A safety gap of at least one epoch length is enforced between partitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]


@dataclass
class TimePartition:
    """Half-open time interval [start, end) in seconds."""
    start: float
    end: float
    split: Split

    @property
    def duration(self) -> float:
        return self.end - self.start


def chronological_time_splits(
    total_duration: float,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    safety_gap: float = 2.0,  # seconds; default = one 2-second epoch
) -> list[TimePartition]:
    """
    Split [0, total_duration) into train/val/test time blocks with safety gaps.

    The safety gap is removed from the *end* of each block so no epoch
    starting near the boundary can overlap into the next partition.

    Parameters
    ----------
    total_duration : float
        Length of the recording in seconds.
    train_frac, val_frac, test_frac : float
        Fractions that must sum to 1.0.
    safety_gap : float
        Seconds excluded at each partition boundary (must be >= epoch_len).

    Returns
    -------
    list[TimePartition]
        Three non-overlapping partitions in chronological order.
    """
    fracs = [train_frac, val_frac, test_frac]
    if not abs(sum(fracs) - 1.0) < 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {sum(fracs):.4f}")

    # Raw boundaries before gap removal
    boundaries = [0.0]
    cumulative = 0.0
    for f in fracs:
        cumulative += f * total_duration
        boundaries.append(cumulative)

    # Apply safety gap: the usable portion of each block ends `safety_gap`
    # seconds before the next block starts, so no epoch straddles the boundary.
    partitions: list[TimePartition] = []
    names: list[Split] = ["train", "val", "test"]
    for i, name in enumerate(names):
        start = boundaries[i]
        end   = boundaries[i + 1] - safety_gap  # shrink the right edge
        if end <= start:
            raise ValueError(
                f"Partition '{name}' has non-positive duration after safety gap "
                f"({end - start:.2f}s). Reduce safety_gap or use longer recordings."
            )
        partitions.append(TimePartition(start=start, end=end, split=name))

    logger.debug(
        "Time splits: train=[%.1f,%.1f) val=[%.1f,%.1f) test=[%.1f,%.1f)",
        partitions[0].start, partitions[0].end,
        partitions[1].start, partitions[1].end,
        partitions[2].start, partitions[2].end,
    )
    return partitions


def epochs_for_partition(
    partition: TimePartition,
    epoch_len: float,
    sfreq: int,
    subject: str,
    task: str,
    recording_id: str = "",
) -> pd.DataFrame:
    """
    Generate non-overlapping epoch metadata for a time partition.

    Does NOT generate signal data — it returns a DataFrame of epoch start/end
    times and metadata.  The actual signal windowing is done in epoching.py.

    Parameters
    ----------
    partition : TimePartition
        The time block to epoch.
    epoch_len : float
        Epoch length in seconds.
    sfreq : int
        Sampling frequency (used only for rounding to sample boundaries).
    subject, task, recording_id : str
        Metadata to stamp on every epoch row.

    Returns
    -------
    pd.DataFrame with columns: subject, task, recording_id, split, start_s, end_s
    """
    starts = np.arange(partition.start, partition.end - epoch_len + 1e-9, epoch_len)
    records = []
    for s in starts:
        e = s + epoch_len
        if e > partition.end + 1e-9:
            break
        records.append({
            "subject":      subject,
            "task":         task,
            "recording_id": recording_id,
            "split":        partition.split,
            "start_s":      round(s, 6),
            "end_s":        round(e, 6),
        })
    df = pd.DataFrame(records)
    logger.debug(
        "%s/%s [%s]: %d non-overlapping epochs in [%.1f, %.1f)",
        subject, task, partition.split, len(df), partition.start, partition.end,
    )
    return df


def check_no_overlap_across_splits(epoch_meta: pd.DataFrame) -> None:
    """
    Assert that no two epochs from different splits share any signal samples.

    Raises AssertionError if any cross-split time overlap is detected.
    """
    for (subj, task), grp in epoch_meta.groupby(["subject", "task"]):
        for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
            a = grp[grp["split"] == split_a]
            b = grp[grp["split"] == split_b]
            for _, ra in a.iterrows():
                for _, rb in b.iterrows():
                    overlap = min(ra["end_s"], rb["end_s"]) - max(ra["start_s"], rb["start_s"])
                    if overlap > 1e-6:
                        raise AssertionError(
                            f"Overlap detected between {split_a} and {split_b} "
                            f"for {subj}/{task}: "
                            f"[{ra['start_s']:.3f},{ra['end_s']:.3f}) ∩ "
                            f"[{rb['start_s']:.3f},{rb['end_s']:.3f})"
                        )


def subject_grouped_folds(
    subjects: list[str],
    n_splits: int | None = None,
) -> list[tuple[list[str], list[str]]]:
    """
    Leave-One-Subject-Out folds for unseen-subject evaluation.

    If `n_splits` is None, returns true LOSO (len(subjects) folds).
    Otherwise returns GroupKFold-style folds.

    Returns
    -------
    list of (train_subjects, test_subjects) tuples
    """
    subjects = list(subjects)
    n = len(subjects)
    if n_splits is None or n_splits >= n:
        # True LOSO
        return [
            ([s for s in subjects if s != test_sub], [test_sub])
            for test_sub in subjects
        ]
    # GroupKFold-style: chunk subjects into n_splits groups
    chunk_size = n // n_splits
    folds = []
    for i in range(n_splits):
        start = i * chunk_size
        end   = start + chunk_size if i < n_splits - 1 else n
        test_subs  = subjects[start:end]
        train_subs = [s for s in subjects if s not in test_subs]
        folds.append((train_subs, test_subs))
    return folds
