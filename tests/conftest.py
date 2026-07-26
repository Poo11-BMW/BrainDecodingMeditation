"""
Shared synthetic EEG fixtures for testing.

All fixtures use deterministic seeds and synthetic data so tests run
without downloading the 8 GB real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, EpochConfig, ModelConfig, SplitConfig


# ── Basic configuration fixture ────────────────────────────────────────────────

@pytest.fixture
def test_config() -> Config:
    cfg = Config()
    cfg.epoch  = EpochConfig(sfreq=64, epoch_len=2.0, epoch_overlap=0.0)
    cfg.split  = SplitConfig(train_frac=0.70, val_frac=0.10, test_frac=0.20, safety_gap_epochs=1)
    cfg.model  = ModelConfig(random_seed=42, n_estimators=10)
    return cfg


# ── Synthetic raw EEG-like signal ─────────────────────────────────────────────

@pytest.fixture
def synthetic_raw_signal():
    """
    Returns a (n_channels, n_times) numpy array simulating 120 s of EEG at 64 Hz.
    16 channels, all standard normally distributed with a 10 Hz sine added to ch-0.
    """
    rng = np.random.default_rng(42)
    sfreq  = 64
    duration = 120.0  # seconds
    n_times = int(duration * sfreq)
    n_ch    = 16
    data = rng.standard_normal((n_ch, n_times))
    t    = np.arange(n_times) / sfreq
    data[0] += 2.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha in ch-0
    return data, sfreq, duration


@pytest.fixture
def synthetic_feature_df():
    """
    Returns a synthetic feature DataFrame with 4 subjects, 2 tasks, and a `split` column.
    Each subject has ~200 epochs split 70/10/20 chronologically.
    """
    rng = np.random.default_rng(42)
    subjects = ["sub-001", "sub-002", "sub-003", "sub-004"]
    tasks    = ["med1breath", "think1"]
    feature_names = [
        "frontal_Delta", "frontal_Theta", "frontal_Alpha", "frontal_Beta", "frontal_Gamma",
        "parietal_Delta", "parietal_Theta", "parietal_Alpha", "parietal_Beta", "parietal_Gamma",
        "global_Delta", "global_Theta", "global_Alpha", "global_Beta", "global_Gamma",
        "ratio_ThetaBeta", "ratio_AlphaBeta", "ratio_GammaBeta",
        "FAA", "Fz_Theta",
        "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
        "PermEntropy",
        "PLV_FP_Alpha", "PLV_FP_Theta",
    ]
    rows = []
    for sub in subjects:
        for task in tasks:
            n_epochs = 100
            n_train = int(n_epochs * 0.70)
            n_val   = int(n_epochs * 0.10)
            n_test  = n_epochs - n_train - n_val
            splits = (["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
            label = 0 if "med" in task else 1
            for sp_label in splits:
                feat_vals = rng.standard_normal(len(feature_names))
                if label == 0:
                    feat_vals[feature_names.index("PLV_FP_Alpha")] += 0.5
                row = dict(zip(feature_names, feat_vals))
                row["Subject"] = sub
                row["Task"]    = task
                row["binary"]  = label
                row["split"]   = sp_label
                rows.append(row)
    df = pd.DataFrame(rows)
    # PLV must be in [0,1]
    for col in ["PLV_FP_Alpha", "PLV_FP_Theta"]:
        df[col] = df[col].clip(0, 1)
    return df, feature_names
