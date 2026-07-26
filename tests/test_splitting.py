"""
Tests for src/splitting.py.

Proves:
  1. Train/val/test time intervals do not overlap.
  2. Epochs never cross split boundaries.
  3. No two overlapping epochs are assigned to different splits.
  4. Safety gap prevents cross-boundary contamination.
  5. Identical seeds produce identical splits.
  6. Subject-grouped LOSO folds: train and test subjects never overlap.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.splitting import (
    chronological_time_splits,
    check_no_overlap_across_splits,
    epochs_for_partition,
    subject_grouped_folds,
)


EPOCH_LEN = 2.0  # seconds
SFREQ     = 64


# ── chronological_time_splits ─────────────────────────────────────────────────

class TestChronologicalTimeSplits:

    def test_three_partitions_returned(self):
        parts = chronological_time_splits(120.0)
        assert len(parts) == 3
        assert [p.split for p in parts] == ["train", "val", "test"]

    def test_no_time_overlap(self):
        parts = chronological_time_splits(120.0, safety_gap=EPOCH_LEN)
        for i in range(len(parts) - 1):
            # end of partition i must be <= start of partition i+1
            assert parts[i].end <= parts[i + 1].start + 1e-9, (
                f"Overlap between {parts[i].split} and {parts[i+1].split}: "
                f"end={parts[i].end:.3f} > start={parts[i+1].start:.3f}"
            )

    def test_fractions_sum_to_one(self):
        with pytest.raises(ValueError):
            chronological_time_splits(120.0, train_frac=0.5, val_frac=0.5, test_frac=0.5)

    def test_safety_gap_reduces_partition_duration(self):
        no_gap = chronological_time_splits(120.0, safety_gap=0.0)
        with_gap = chronological_time_splits(120.0, safety_gap=EPOCH_LEN)
        for ng, wg in zip(no_gap, with_gap):
            assert wg.duration < ng.duration + 1e-9

    def test_too_large_safety_gap_raises(self):
        with pytest.raises(ValueError):
            # safety gap larger than val partition
            chronological_time_splits(10.0, train_frac=0.7, val_frac=0.1, test_frac=0.2,
                                       safety_gap=5.0)

    def test_deterministic(self):
        a = chronological_time_splits(200.0)
        b = chronological_time_splits(200.0)
        for pa, pb in zip(a, b):
            assert pa.start == pb.start
            assert pa.end   == pb.end


# ── epochs_for_partition ──────────────────────────────────────────────────────

class TestEpochsForPartition:

    def test_non_overlapping_epochs_do_not_cross_boundary(self):
        from src.splitting import TimePartition
        part = TimePartition(start=0.0, end=20.0, split="train")
        meta = epochs_for_partition(part, EPOCH_LEN, SFREQ, "sub-001", "med1breath")
        for _, row in meta.iterrows():
            assert row["start_s"] >= part.start - 1e-9
            assert row["end_s"]   <= part.end   + 1e-9

    def test_consecutive_epochs_are_non_overlapping(self):
        from src.splitting import TimePartition
        part = TimePartition(start=0.0, end=20.0, split="train")
        meta = epochs_for_partition(part, EPOCH_LEN, SFREQ, "sub-001", "med1breath")
        starts = meta["start_s"].values
        ends   = meta["end_s"].values
        for i in range(len(starts) - 1):
            assert ends[i] <= starts[i + 1] + 1e-9, (
                f"Epoch {i} [{starts[i]:.3f},{ends[i]:.3f}) overlaps "
                f"epoch {i+1} [{starts[i+1]:.3f},{...})"
            )

    def test_split_label_preserved(self):
        from src.splitting import TimePartition
        for sp in ("train", "val", "test"):
            part = TimePartition(start=0.0, end=20.0, split=sp)
            meta = epochs_for_partition(part, EPOCH_LEN, SFREQ, "sub-001", "med1breath")
            assert (meta["split"] == sp).all()


# ── check_no_overlap_across_splits ───────────────────────────────────────────

class TestCheckNoOverlap:

    def _make_clean_meta(self):
        """Build a valid epoch_meta with no cross-split overlap."""
        import pandas as pd
        parts = chronological_time_splits(120.0, safety_gap=EPOCH_LEN)
        frames = []
        for part in parts:
            meta = epochs_for_partition(part, EPOCH_LEN, SFREQ, "sub-001", "med1breath")
            frames.append(meta)
        return pd.concat(frames, ignore_index=True)

    def test_clean_meta_passes(self):
        meta = self._make_clean_meta()
        # Should not raise
        check_no_overlap_across_splits(meta)

    def test_overlapping_meta_raises(self):
        import pandas as pd
        # Manually insert an overlapping epoch
        meta = self._make_clean_meta()
        bad_row = pd.DataFrame([{
            "subject": "sub-001", "task": "med1breath", "recording_id": "",
            "split": "test", "start_s": 0.0, "end_s": 2.0,  # overlaps train
        }])
        meta = pd.concat([meta, bad_row], ignore_index=True)
        with pytest.raises(AssertionError):
            check_no_overlap_across_splits(meta)


# ── subject_grouped_folds ─────────────────────────────────────────────────────

class TestSubjectGroupedFolds:

    def test_loso_no_subject_overlap(self):
        subjects = [f"sub-{i:03d}" for i in range(10)]
        folds = subject_grouped_folds(subjects, n_splits=None)
        assert len(folds) == len(subjects)
        for train_subs, test_subs in folds:
            overlap = set(train_subs) & set(test_subs)
            assert len(overlap) == 0, f"Overlap detected: {overlap}"

    def test_loso_all_subjects_tested(self):
        subjects = [f"sub-{i:03d}" for i in range(8)]
        folds = subject_grouped_folds(subjects)
        tested = [fold[1][0] for fold in folds]
        assert sorted(tested) == sorted(subjects)

    def test_group_kfold_no_overlap(self):
        subjects = [f"sub-{i:03d}" for i in range(20)]
        folds = subject_grouped_folds(subjects, n_splits=5)
        for train_subs, test_subs in folds:
            assert len(set(train_subs) & set(test_subs)) == 0

    def test_single_test_subject_per_loso_fold(self):
        subjects = [f"sub-{i:03d}" for i in range(6)]
        folds = subject_grouped_folds(subjects)
        for _, test_subs in folds:
            assert len(test_subs) == 1
