"""
Tests for src/epoching.py.

Proves:
  1. Epochs generated within a partition never cross partition boundaries.
  2. Epochs are non-overlapping (step = epoch_len).
  3. Epoch metadata is correctly populated.
  4. Empty recordings return empty arrays, not errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.epoching import build_epoch_metadata, epoch_partition
from src.splitting import TimePartition


SFREQ     = 64
EPOCH_LEN = 2.0


def _make_fake_raw(duration: float, n_ch: int = 8, sfreq: float = SFREQ):
    """Minimal fake MNE-like object using a namespace with get_data() and info."""
    import types
    rng = np.random.default_rng(0)
    n_times = int(duration * sfreq)
    data = rng.standard_normal((n_ch, n_times))

    raw = types.SimpleNamespace()
    raw.get_data = lambda: data
    raw.info     = {"sfreq": sfreq}
    raw.times    = np.arange(n_times) / sfreq
    return raw


class TestEpochPartition:

    def test_epochs_within_partition_boundaries(self):
        raw  = _make_fake_raw(duration=120.0)
        part = TimePartition(start=10.0, end=80.0, split="train")
        epochs, starts = epoch_partition(raw, part, EPOCH_LEN, int(SFREQ))
        for s in starts:
            assert s >= part.start - 1e-9
            assert s + EPOCH_LEN <= part.end + 1e-9

    def test_epochs_non_overlapping(self):
        raw  = _make_fake_raw(duration=60.0)
        part = TimePartition(start=0.0, end=60.0, split="train")
        _, starts = epoch_partition(raw, part, EPOCH_LEN, int(SFREQ))
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - starts[i]
            assert abs(gap - EPOCH_LEN) < 1e-9, f"Unexpected gap {gap} between epochs {i} and {i+1}"

    def test_correct_number_of_epochs(self):
        raw  = _make_fake_raw(duration=20.0)
        part = TimePartition(start=0.0, end=20.0, split="train")
        epochs, starts = epoch_partition(raw, part, EPOCH_LEN, int(SFREQ))
        expected = int((part.end - part.start) / EPOCH_LEN)
        assert len(epochs) == expected
        assert len(starts) == expected

    def test_epoch_shape_is_correct(self):
        n_ch = 12
        raw  = _make_fake_raw(duration=30.0, n_ch=n_ch)
        part = TimePartition(start=0.0, end=20.0, split="train")
        epochs, _ = epoch_partition(raw, part, EPOCH_LEN, int(SFREQ))
        expected_times = int(EPOCH_LEN * SFREQ)
        assert epochs.shape == (len(epochs), n_ch, expected_times)

    def test_short_partition_returns_empty(self):
        raw  = _make_fake_raw(duration=1.0)
        # Partition shorter than epoch_len
        part = TimePartition(start=0.0, end=1.0, split="test")
        epochs, starts = epoch_partition(raw, part, EPOCH_LEN, int(SFREQ))
        assert len(epochs) == 0
        assert len(starts) == 0

    def test_sfreq_mismatch_raises(self):
        raw = _make_fake_raw(duration=20.0, sfreq=64)
        part = TimePartition(start=0.0, end=20.0, split="train")
        with pytest.raises(ValueError, match="sfreq"):
            epoch_partition(raw, part, EPOCH_LEN, sfreq=128)

    def test_safety_gap_prevents_boundary_epochs(self):
        """Epochs must stop before partition.end = recording_end - safety_gap."""
        raw  = _make_fake_raw(duration=100.0)
        safety_gap = EPOCH_LEN
        # Partition ends 2 s before the recording end
        part = TimePartition(start=0.0, end=100.0 - safety_gap, split="train")
        _, starts = epoch_partition(raw, part, EPOCH_LEN, int(SFREQ))
        if starts:
            assert max(starts) + EPOCH_LEN <= part.end + 1e-9


class TestBuildEpochMetadata:

    def test_metadata_columns(self):
        part   = TimePartition(start=0.0, end=20.0, split="train")
        starts = [0.0, 2.0, 4.0]
        meta   = build_epoch_metadata(starts, part, EPOCH_LEN, "sub-001", "med1breath", "rec_01")
        assert set(["subject", "task", "recording_id", "split", "start_s", "end_s"]).issubset(meta.columns)

    def test_end_equals_start_plus_epoch_len(self):
        part   = TimePartition(start=0.0, end=20.0, split="val")
        starts = [0.0, 2.0, 4.0, 6.0]
        meta   = build_epoch_metadata(starts, part, EPOCH_LEN, "sub-002", "think1")
        for _, row in meta.iterrows():
            assert abs(row["end_s"] - row["start_s"] - EPOCH_LEN) < 1e-9

    def test_split_label_set_correctly(self):
        part = TimePartition(start=0.0, end=20.0, split="test")
        meta = build_epoch_metadata([0.0, 2.0], part, EPOCH_LEN, "sub-003", "think2")
        assert (meta["split"] == "test").all()
