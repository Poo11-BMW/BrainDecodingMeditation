"""
Tests for src/feature_extraction.py.

Proves:
  1. PLV values remain in [0, 1].
  2. Feature extraction returns expected column names.
  3. Missing EEG channels are handled safely (NaN returned, no crash).
  4. Hjorth complexity is non-negative.
  5. Permutation entropy is non-negative.
  6. Feature extraction is deterministic (same input → same output).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_extraction import (
    compute_plv,
    extract_features,
    get_feature_groups,
    hjorth,
    perm_entropy,
)


SFREQ = 64
N_TIMES = 128   # 2 seconds at 64 Hz
N_CH    = 16

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta":  (12.0, 30.0),
    "Gamma": (30.0, 40.0),
}

REGION_DEFS = {
    "frontal":   ["FP1", "FP2", "F3", "FZ", "F4"],
    "central":   ["C3", "CZ", "C4"],
    "temporal":  ["T7", "T8"],
    "parietal":  ["P3", "PZ", "P4"],
    "occipital": ["O1", "OZ", "O2"],
}

CH_NAMES = ["FP1", "FP2", "F3", "FZ", "F4", "C3", "CZ", "C4",
            "T7",  "T8",  "P3", "PZ", "P4", "O1", "OZ", "O2"]

assert len(CH_NAMES) == N_CH


def _make_epoch(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_CH, N_TIMES)).astype(float)


def _make_psd(epoch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch
    freqs, psd = welch(epoch, fs=SFREQ, nperseg=64)
    return psd, freqs


class TestPLV:

    def test_plv_in_valid_range(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal((4, N_TIMES))
            b = rng.standard_normal((4, N_TIMES))
            plv = compute_plv(a, b)
            assert 0.0 <= plv <= 1.0, f"PLV out of range: {plv}"

    def test_identical_signals_have_high_plv(self):
        sig = np.sin(2 * np.pi * 10 * np.arange(N_TIMES) / SFREQ)
        a = np.tile(sig, (4, 1))
        b = np.tile(sig, (3, 1))
        plv = compute_plv(a, b)
        assert plv > 0.9, f"Expected high PLV for identical signals, got {plv}"

    def test_random_signals_have_low_plv(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8, 1024))
        b = rng.standard_normal((8, 1024))
        plv = compute_plv(a, b)
        assert plv < 0.5, f"Expected low PLV for random signals, got {plv}"


class TestHjorth:

    def test_activity_is_non_negative(self):
        epoch = _make_epoch()
        act, mob, comp = hjorth(epoch)
        assert act >= 0.0

    def test_complexity_is_non_negative(self):
        epoch = _make_epoch()
        act, mob, comp = hjorth(epoch)
        assert comp >= 0.0

    def test_constant_signal_has_zero_mobility(self):
        data = np.ones((4, N_TIMES))
        _, mob, _ = hjorth(data)
        assert abs(mob) < 1e-6


class TestPermEntropy:

    def test_entropy_non_negative(self):
        epoch = _make_epoch()
        pe = perm_entropy(epoch)
        assert pe >= 0.0

    def test_constant_signal_has_zero_entropy(self):
        data = np.ones((2, N_TIMES))
        pe = perm_entropy(data)
        assert pe == pytest.approx(0.0, abs=1e-6)


class TestExtractFeatures:

    def _run_extraction(self, seed: int = 0) -> dict:
        epoch = _make_epoch(seed)
        psd, freqs = _make_psd(epoch)
        return extract_features(
            epoch_data=epoch,
            psd_data=psd,
            freqs=freqs,
            ch_names=CH_NAMES,
            region_defs=REGION_DEFS,
            bands=BANDS,
            sfreq=SFREQ,
        )

    def test_expected_feature_names_present(self):
        feat = self._run_extraction()
        expected = [
            "PLV_FP_Alpha", "PLV_FP_Theta",
            "FAA", "Fz_Theta",
            "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
            "PermEntropy",
            "global_Alpha", "global_Theta",
            "ratio_ThetaBeta", "ratio_AlphaBeta",
            "frontal_Alpha", "parietal_Alpha",
        ]
        for name in expected:
            assert name in feat, f"Missing feature: {name}"

    def test_plv_features_in_valid_range(self):
        feat = self._run_extraction()
        for key in ("PLV_FP_Alpha", "PLV_FP_Theta"):
            assert 0.0 <= feat[key] <= 1.0, f"{key}={feat[key]} out of [0,1]"

    def test_deterministic_with_same_input(self):
        feat_a = self._run_extraction(seed=123)
        feat_b = self._run_extraction(seed=123)
        for k in feat_a:
            if isinstance(feat_a[k], float) and not np.isnan(feat_a[k]):
                assert feat_a[k] == pytest.approx(feat_b[k], rel=1e-6)

    def test_missing_channel_returns_nan_not_error(self):
        epoch  = _make_epoch()
        psd, freqs = _make_psd(epoch)
        # Use channel names that don't include F3/F4 (needed for FAA)
        partial_chs = [c for c in CH_NAMES if c not in ("F3", "F4")]
        epoch_sub = epoch[[i for i, c in enumerate(CH_NAMES) if c in partial_chs]]
        psd_sub   = psd[[i for i, c in enumerate(CH_NAMES) if c in partial_chs]]
        feat = extract_features(
            epoch_data=epoch_sub,
            psd_data=psd_sub,
            freqs=freqs,
            ch_names=partial_chs,
            region_defs=REGION_DEFS,
            bands=BANDS,
            sfreq=SFREQ,
        )
        assert np.isnan(feat["FAA"]), "FAA should be NaN when F3/F4 are missing"

    def test_total_feature_count(self):
        feat = self._run_extraction()
        # Must have at least 41 features (the documented count)
        assert len(feat) >= 41, f"Expected >= 41 features, got {len(feat)}"


class TestGetFeatureGroups:

    def test_all_features_assigned_to_a_group(self):
        feat = TestExtractFeatures()._run_extraction()
        feat_cols = list(feat.keys())
        groups = get_feature_groups(feat_cols)
        assigned = set(col for cols in groups.values() for col in cols)
        for col in feat_cols:
            assert col in assigned, f"Feature '{col}' not assigned to any group"

    def test_plv_features_in_connectivity_group(self):
        feat = TestExtractFeatures()._run_extraction()
        groups = get_feature_groups(list(feat.keys()))
        assert "PLV_FP_Alpha" in groups.get("connectivity_plv", [])
        assert "PLV_FP_Theta" in groups.get("connectivity_plv", [])

    def test_hjorth_in_hjorth_group(self):
        feat = TestExtractFeatures()._run_extraction()
        groups = get_feature_groups(list(feat.keys()))
        assert "Hjorth_Complexity" in groups.get("hjorth", [])
